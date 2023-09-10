import pygame
import numpy as np
from scipy.io import wavfile
import cv2 as cv
import PySpectralysis
import time
import tkinter as tk
from tkinter import filedialog
import threading

pygame.mixer.pre_init(44100//2, size=-16, channels=1)
pygame.init()

root = tk.Tk()
root.withdraw()

audio_path = filedialog.askopenfilename(filetypes=[("WAV Audio File", ".wav")])

print(audio_path)

CHUNK_SIZE = 32
inv_chunks = {}

SPEC_HEIGHT = 1024*8
specsis = PySpectralysis.Spectralysis(SPEC_HEIGHT // 4, SPEC_HEIGHT)
(in_len, spec_size, out_len, _) = specsis.getsize()
spec_height = in_len - out_len
signal_pad = spec_height // 2

samplerate, audiodata = wavfile.read(audio_path)
if len(audiodata.shape) == 2:
    audiodata = audiodata[:, 0]
audiodata = audiodata / np.max(np.abs(audiodata))
filtdata = audiodata.copy()[signal_pad:-signal_pad]
filtdata = filtdata.astype(np.float32)
audiodata = audiodata.astype(np.float32)

result_signal = np.zeros_like(audiodata)
nchunks = (len(audiodata) - signal_pad) // out_len
winsize = (800, 600)
window = pygame.display.set_mode(winsize)
pygame.display.set_icon(pygame.image.load('icon_sm.png'))


def getBrush(size, blur):
    img = np.zeros((size*4, size*4))
    img = cv.circle(img, (img.shape[0] // 2, img.shape[1] // 2), size // 2, 255, -1)
    img = cv.blur(img, (blur, blur))
    result = pygame.Surface(img.shape, pygame.SRCALPHA, 32).convert_alpha()
    result.fill((255, 255, 255, 0))
    alphas = pygame.surfarray.pixels_alpha(result)
    alphas[:, :] = img.astype(np.uint8)

    return result

class Pannable:
    def __init__(self, dstrect, srcrect, nchunks):
        super().__init__()
        # srcrect[3] = srcrect[3] // 2
        self.chunkwidth = srcrect[2] // nchunks
        srcrect[2] = self.chunkwidth * nchunks
        self.dstrect = dstrect
        self.srcrect = srcrect
        self.size = self.dstrect[2], self.dstrect[3]
        self.srcsize = self.srcrect[2], self.srcrect[3]
        self.bg = pygame.surfarray.make_surface(np.ones((self.srcrect[2], self.srcrect[3], 3)) * 255)
        self.fg = pygame.Surface((self.srcrect[2], self.srcrect[3]), pygame.SRCALPHA, 32).convert_alpha()
        self.fg.fill((0, 0, 0, 0))
        self.bgrect = self.bg.get_rect()

        self.viewport = srcrect.copy()
        self.viewport.height = self.viewport.height // 2  # Only real part of the spectrogram will be shown
        self.zoomlevel = 1
        self.zoomlevel_x = self.dstrect.size[0] / self.srcrect.size[0]
        self.zoomlevel_y = self.dstrect.size[1] / self.srcrect.size[1]

        self.dx = 0
        self.dy = 0
        self.mouse_dx = 0
        self.mouse_dy = 0

        self.blitmap()
        pygame.display.flip()

    def set_bg(self, chunk, data):
        print(chunk, chunk * self.chunkwidth, data.shape, np.max(data), np.min(data))
        data = data * 255
        surf = pygame.surfarray.make_surface(np.stack([data, data, data], axis=-1).astype(np.uint8))
        self.bg.blit(surf, (chunk * self.chunkwidth, 0))

    def blitmap(self, window=None):
        bg_cropped = pygame.Surface(self.viewport.size)
        bg_cropped.blit(self.bg, (0, 0), self.viewport)
        bg_cropped.blit(self.fg, (0, 0), self.viewport)
        bgsurface = pygame.transform.smoothscale(bg_cropped, self.size)

        if window is not None:
            window.blit(bgsurface, (self.dstrect[0], self.dstrect[1]))

    def zoom(self, zoom, pos):
        if self.viewport.size[0] / zoom > self.srcrect.width or self.viewport.size[1] / zoom > self.srcrect.height // 2:
            if self.viewport.size[0] / zoom > self.srcrect.width:
                self.viewport.left = 0
                self.viewport.width = self.srcrect.width
            if self.viewport.size[1] / zoom > self.srcrect.height // 2:
                self.viewport.top = 0
                self.viewport.height = self.srcrect.height // 2
            return

        mx, my = pos
        mx = mx / self.size[0] * self.viewport.size[0]
        my = my / self.size[1] * self.viewport.size[1]
        left = self.viewport.left - mx / zoom + mx
        top = self.viewport.top - my / zoom + my
        self.zoomlevel *= zoom
        self.viewport = pygame.Rect(left, top,
                                    self.viewport.size[0] / zoom, self.viewport.size[1] / zoom)

        if self.viewport.left < 0:
            self.viewport.left = 0
        if self.viewport.top < 0:
            self.viewport.top = 0
        if self.viewport.right > self.srcrect.width:
            self.viewport.right = self.srcrect.width
        if self.viewport.bottom > self.srcrect.height // 2:
            self.viewport.bottom = self.srcrect.height // 2


    def pan(self, dx, dy):
        self.dx += dx / self.zoomlevel_x / self.zoomlevel
        self.dy += dy / self.zoomlevel_y / self.zoomlevel
        dx = self.dx // 1
        dy = self.dy // 1
        self.dx -= dx
        self.dy -= dy
        self.viewport.move_ip(-dx, -dy)
        if self.viewport.left < 0:
            self.viewport.left = 0
        if self.viewport.top < 0:
            self.viewport.top = 0
        if self.viewport.right > self.srcrect.width:
            self.viewport.right = self.srcrect.width
        if self.viewport.bottom > self.srcrect.height // 2:
            self.viewport.bottom = self.srcrect.height // 2

        # self.blitmap()

class Drawer(Pannable):
    def __init__(self, dstrect, srcrect, nchunks):
        super().__init__(dstrect, srcrect, nchunks)
        self.brushSize = 10
        self.brushBlur = 2
        self.buildBrush()

    def buildBrush(self):
        if self.brushBlur % 2 == 0:
            self.brushBlur += 1
        self.brush = getBrush(self.brushSize, self.brushBlur)
        self.brush_rect = self.brush.get_rect()

    def draw_single(self, x, y):
        zoomlevel = self.zoomlevel * max(self.zoomlevel_x, self.zoomlevel_y)
        brushsurface = pygame.transform.smoothscale(self.brush, (
            self.viewport[2] / self.size[0] * self.brush_rect[2] * zoomlevel,
            self.viewport[3] / self.size[1] * self.brush_rect[3] * zoomlevel))
        xstart = x / self.size[0] * self.viewport.width - brushsurface.get_rect().size[0] // 2
        ystart = y / self.size[1] * self.viewport.height - brushsurface.get_rect().size[1] // 2
        self.fg.blit(brushsurface, (
            xstart + self.viewport.left,
            ystart + self.viewport.top
        ))
        chunk_start = int(xstart + self.viewport.left) // self.chunkwidth
        chunk_end = max(int(xstart + self.brush_rect.size[0] + self.viewport.left) // self.chunkwidth, chunk_start + 1)
        # print(chunk_start, chunk_end, xstart, self.brush_rect.size[0], brushsurface.get_rect().size[0], self.chunkwidth)
        # print(int(xstart + self.viewport.left), int(xstart + self.viewport.left) // self.chunkwidth)
        for c in range(chunk_start, chunk_end+1):
            # print(c)
            if c < 0 or c > nchunks - 1:
                continue
            inv_chunks[c] = 1

    def update(self, x, y, window):
        self.blitmap(window)
        zoomlevel = self.zoomlevel * max(self.zoomlevel_x, self.zoomlevel_y)
        newsize = (int(self.brush_rect.size[0] * zoomlevel), int(self.brush_rect.size[1] * zoomlevel))
        brush = pygame.transform.smoothscale(self.brush, newsize)
        window.blit(brush, (x - newsize[0] // 2, y - newsize[1] // 2))
        pygame.display.update()

class Speaker(Pannable):
    def __init__(self, dstrect, srcrect, nchunks):
        super().__init__(dstrect, srcrect, nchunks)
        self.range = (0, nchunks * self.chunkwidth)
        self.updateRange()

    def updateRange(self):
        self.fg.fill((0, 0, 0, 0))
        self.fg.fill((0, 0, 255, 64), (self.range[0], 0, self.range[1] - self.range[0], self.srcrect[3]))

    def setRangeEnd(self, x):
        x = x / self.dstrect[2] * self.viewport[2] + self.viewport.left
        self.range = (min(x, self.range[0]), max(x, self.range[0]))
        self.updateRange()

    def setRangeStart(self, x):
        x = x / self.dstrect[2] * self.viewport[2] + self.viewport.left
        self.range = (x, x)
        self.updateRange()


drawer = Drawer(pygame.Rect(0, winsize[1] // 2, winsize[0], winsize[1] // 2), pygame.Rect(0, 0, 32*nchunks, SPEC_HEIGHT), nchunks)
speaker = Speaker(pygame.Rect(0, 0, winsize[0], winsize[1] // 2), pygame.Rect(0, 0, 32*nchunks, SPEC_HEIGHT), nchunks)

# Calculate initial spectrogram
output = np.zeros(out_len, dtype=float)
out_spec = np.zeros(spec_size, dtype=float)
for chunk in range(nchunks):
    start = signal_pad + chunk * out_len
    src_signal = audiodata[start:start + out_len]
    print(src_signal.shape, out_len)
    spec = specsis.sdft(src_signal).reshape(-1, spec_height) / spec_height * 2000
    spec = np.log(spec + 1) / np.log(100)
    spec[spec > 1] = 1
    drawer.set_bg(chunk, spec)
    filt_signal = filtdata[chunk * out_len:(chunk + 1) * out_len]
    print(filt_signal.shape, out_len)
    if len(filt_signal) < out_len:
        continue
    spec = specsis.sdft(filt_signal).reshape(-1, spec_height) / spec_height * 2000
    spec = np.log(spec + 1) / np.log(100)
    spec[spec > 1] = 1
    speaker.set_bg(chunk, spec)

drawer.blitmap(window)
speaker.blitmap(window)

last_time = time.time()
def filter_chunk(chunk):
    global last_time
    signal_start = out_len * chunk
    signal_end = signal_start + in_len
    signal = audiodata[signal_start:signal_end]

    masksurf = pygame.Surface((drawer.chunkwidth, drawer.srcsize[1] // 2), pygame.SRCALPHA, 32)
    masksurf.blit(drawer.fg, (0, 0), (chunk * drawer.chunkwidth, 0, drawer.chunkwidth, drawer.srcsize[1] // 2))
    mask = pygame.surfarray.array2d(masksurf)
    mask = np.right_shift(np.bitwise_and(mask, 0xff000000), 24)
    mask = np.concatenate([mask[:, ::-1], mask], axis=-1)

    print('In between', time.time() - last_time)
    last_time = time.time()

    filt_signal = specsis.process(signal, 255 - mask)

    print('Processing', time.time() - last_time)
    last_time = time.time()

    spec = specsis.sdft(filt_signal).reshape(-1, spec_height) / spec_height * 2000
    spec = np.log(spec + 1) / np.log(100)
    spec[spec > 1] = 1

    print('SDFT', time.time() - last_time)
    last_time = time.time()

    print('')

    filtdata[signal_start:signal_start + out_len] = filt_signal.astype(np.float32)
    speaker.set_bg(chunk, spec)
    speaker.blitmap(window)

def showProgress(window):
    # print(window, window.get_size())
    winwidth = window.get_size()[0]
    progress_colors = np.zeros((winwidth, 1, 3))
    progress_colors[:, :, 1] = 255
    chunkwidth = winwidth / nchunks
    for c in inv_chunks:
        chunk_colors = progress_colors[int(chunkwidth * c):int(chunkwidth * (c + 1)), :]
        chunk_colors[:, :, 1] = 0
        chunk_colors[:, :, 0] = 255
    progress = pygame.surfarray.make_surface(progress_colors)
    window.blit(progress, (0, 1))

is_running = True
is_pan = False
is_draw = False

trigger = threading.Event()
def filtererThread(trigger):
    global inv_chunks, is_running
    while is_running:
        if trigger.wait(0.1):
            for c in inv_chunks:
                filter_chunk(c)
                del inv_chunks[c]
                break
            if len(inv_chunks) == 0:
                trigger.clear()
threading.Thread(target=filtererThread, args=[trigger]).start()
while is_running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            print(pygame.mixer.get_busy())
            if pygame.mixer.get_busy():
                pygame.mixer.stop()
            else:
                buffer = filtdata[int(speaker.range[0] / speaker.chunkwidth * out_len)
                                  :int(speaker.range[1] / speaker.chunkwidth * out_len)]
                sound = pygame.mixer.Sound(buffer=(buffer * 10000).astype(np.int16))
                sound.play()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == pygame.BUTTON_WHEELUP or event.button == pygame.BUTTON_WHEELDOWN:
                pressed = pygame.key.get_pressed()
                x, y = pygame.mouse.get_pos()
                if event.pos[1] > drawer.dstrect[1]:
                    # print(pressed[pygame.K_LALT], pressed[pygame.K_LCTRL])
                    if pressed[pygame.K_LCTRL] and pressed[pygame.K_LSHIFT]:
                        drawer.brushBlur += 2 if event.button == pygame.BUTTON_WHEELUP else -2
                        if drawer.brushBlur < 1:
                            drawer.brushBlur = 1
                        drawer.buildBrush()
                        drawer.update(x, y, window)
                    elif pressed[pygame.K_LCTRL]:
                        drawer.brushSize += 1 if event.button == pygame.BUTTON_WHEELUP else -1
                        if drawer.brushSize < 1:
                            drawer.brushSize = 1
                        drawer.buildBrush()
                        drawer.update(x, y, window)
                    else:
                        pos = (event.pos[0], event.pos[1] - drawer.dstrect[1])
                        zoom = 1.2 if event.button == pygame.BUTTON_WHEELUP else 0.8
                        drawer.zoom(zoom, pos)
                        drawer.blitmap(window)
                else:
                    zoom = 1.2 if event.button == pygame.BUTTON_WHEELUP else 0.8
                    speaker.zoom(zoom, event.pos)
                    speaker.blitmap(window)
            if event.button == pygame.BUTTON_MIDDLE:
                pygame.mouse.get_rel()
                is_pan = True
            if event.button == pygame.BUTTON_LEFT:
                is_draw = True
                if event.pos[1] < drawer.dstrect[1]:
                    speaker.setRangeStart(x)
                else:
                    drawer.draw_single(x, y)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == pygame.BUTTON_MIDDLE:
            is_pan = False
        elif event.type == pygame.MOUSEBUTTONUP and event.button == pygame.BUTTON_LEFT:
            is_draw = False
            drawer.mouse_dx = 0
            drawer.mouse_dy = 0
            trigger.set()
        if event.type == pygame.MOUSEMOTION and is_pan:
            dx, dy = pygame.mouse.get_rel()
            x, y = pygame.mouse.get_pos()
            if y > drawer.dstrect[1]:
                drawer.pan(dx, dy)
            else:
                speaker.pan(dx, dy)
            drawer.blitmap(window)
        if event.type == pygame.MOUSEMOTION:
            x, y = pygame.mouse.get_pos()
            if y > drawer.dstrect[1]:
                drawer.update(x, y, window)
                y = y - drawer.dstrect[1]

                if is_draw:
                    drawer.draw_single(x, y)
                    print(inv_chunks)

                pygame.display.update()
            else:
                if is_draw:
                    speaker.setRangeEnd(x)
                speaker.blitmap(window)
    showProgress(window)
    pygame.display.update()
