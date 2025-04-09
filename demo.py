import marimo

__generated_with = "0.12.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import av
    from tqdm import tqdm
    import marimo as mo
    import matplotlib as plt
    return av, mo, plt, tqdm


@app.cell
def _(av, np, tqdm):
    def calc_brightness():

        with av.open("BigBuckBunny_320x180.mp4") as f:
            brightness = []
            for frame in tqdm(f.decode(video=0), total=f.streams.video[0].frames):
                im = frame.to_image()
                im_lum = im.convert("L")
                arr = np.array(im_lum)
                brightness.append(np.mean(arr) / 255)

            return brightness

    brightness = calc_brightness()
    return brightness, calc_brightness


@app.cell
def _(tqdm):
    import time

    def demo():
        for i in tqdm(list(range(10))):
            time.sleep(1)

    demo()
    return demo, time


@app.cell
def _():
    from PIL import Image
    import numpy as np

    def pil_demo():
        im = Image.open("images/013.png")
        im_lum = im.convert("L")
        arr = np.array(im_lum)
        print(np.mean(arr) / 255)
        return im_lum

    pil_demo()
    return Image, np, pil_demo


@app.cell
def _(brightness, plt):
    from matplotlib.pyplot import figure
    figure(figsize=(16, 4))
    plt.pyplot.plot(brightness)
    return (figure,)


if __name__ == "__main__":
    app.run()
