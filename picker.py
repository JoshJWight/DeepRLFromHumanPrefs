import threading
from enum import Enum
import gi
import numpy as np

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GdkPixbuf
from gi.repository import GLib
from gi.repository import GObject

def pixbuf_from_array(z):
    z=z.astype('uint8')
    h,w,c = z.shape
    if not(c==3 or c==4):
        print(f"Bad shape for pixbuf: {z.shape}")
        assert(False)
    Z = GLib.Bytes.new(z.tobytes())
    useAlpha = (c==4)
    pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, useAlpha, 8, w, h, w*c)
    pixbuf = pixbuf.scale_simple(336, 336, GdkPixbuf.InterpType.NEAREST)
    return pixbuf

class PickerResult(Enum):
    LEFT=1
    RIGHT=2
    SAME=3
    DISCARD=4

class PickerWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Clip Picker")

        #Constants
        self.timeout = 100
        self.postblackout = 2
        #End Constants
        
        self.judgingClip = False
        self.hasResult = False

        topBox = Gtk.Box(spacing=6)
        bottomBox = Gtk.Box(spacing=6)

        leftImage = Gtk.Image()
        rightImage = Gtk.Image()
        
        topBox.pack_start(leftImage, True, True, 0)
        topBox.pack_start(rightImage, True, True, 0)

        self.imageViews = [leftImage, rightImage]

        leftButton = Gtk.Button(label="Left")
        leftButton.connect("clicked", self.onLeftClicked)
        bottomBox.pack_start(leftButton, True, True, 0)

        sameButton = Gtk.Button(label="Same")
        sameButton.connect("clicked", self.onSameClicked)
        bottomBox.pack_start(sameButton, True, True, 0)

        discardButton = Gtk.Button(label="Discard")
        discardButton.connect("clicked", self.onDiscardClicked)
        bottomBox.pack_start(discardButton, True, True, 0)

        rightButton = Gtk.Button(label="Right")
        rightButton.connect("clicked", self.onRightClicked)
        bottomBox.pack_start(rightButton, True, True, 0)

        mainBox = Gtk.VBox(spacing=6)

        mainBox.pack_start(topBox, True, True, 0)
        mainBox.pack_start(bottomBox, True, True, 0)

        self.add(mainBox)

        self.animFrame = [0, 0]
        self.genDefaultClips()
        self.animStep()

    def gtkMain(self):
        self.gtkThread = threading.Thread(target=Gtk.main)
        self.gtkThread.start()

    def getResult(self):
        return (self.clipsnp, self.result)
        
    def genDefaultClips(self):
        h = 80
        w = 80
        c = 3

        self.clips=[]
        for i in range(2):
            clip = []
            for n in range(50):
                buf = np.random.randint(256, size=(h, w, c), dtype='uint8')
                clip.append(pixbuf_from_array(buf))
            self.clips.append(clip)
        
    def setClips(self, clipsnp):
        frameshape = (84, 84, 3)#TODO magic numbers

        self.clipsnp = clipsnp
        clips = []
        for clipnp in clipsnp:
            clip=[]
            for frame in clipnp:
                #fixedFrame = np.moveaxis(frame, 0, 2)
                fixedFrame = np.zeros(frameshape, dtype='uint8')
                for i in range(3):
                    fixedFrame[:84, :84, i] = frame[3, :84, :84]
                clip.append(pixbuf_from_array(fixedFrame))
            for i in range(self.postblackout):
                clip.append(pixbuf_from_array(np.zeros(frameshape, dtype='uint8')))
            
            clips.append(clip)
        self.clips = clips
        self.animFrame = [0, 0]
        self.judgingClip = True
        self.hasResult = False

    def animStep(self):
        if self.judgingClip:
            for i in range(2):
                self.imageViews[i].set_from_pixbuf(self.clips[i][self.animFrame[i]])
                self.animFrame[i] = (self.animFrame[i] + 1) % len(self.clips[i])
        GObject.timeout_add(self.timeout, self.animStep)

    def pickResult(self, result):
        if self.judgingClip:
            self.result = result
            self.judgingClip = False
            self.hasResult = True
            print(f"Result: {self.result}")
        else:
            print("No clip to pick result for")

    def onLeftClicked(self, widget):
        self.pickResult(PickerResult.LEFT)
        print("Left")

    def onSameClicked(self, widget):
        self.pickResult(PickerResult.SAME)
        print("Same")

    def onDiscardClicked(self, widget):
        self.pickResult(PickerResult.DISCARD)
        print("Discard")

    def onRightClicked(self, widget):
        self.pickResult(PickerResult.RIGHT)
        print("Right")


#win = PickerWindow()
#win.connect("destroy", Gtk.main_quit)
#win.show_all()
#Gtk.main()
