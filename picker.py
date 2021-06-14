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
    assert c==3 or c==4
    Z = GLib.Bytes.new(z.tobytes())
    useAlpha = (c==4)
    return GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, useAlpha, 8, w, h, w*c)

class PickerResult(Enum):
    LEFT=1
    RIGHT=2
    SAME=3
    DISCARD=4

class PickerWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Clip Picker")

        self.judgingClip = True #TODO

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

        self.animFrame = 0
        self.genDefaultClips()
        self.animStep()

        
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
        

    def animStep(self):
        if not(self.animFrame==0 and not self.judgingClip):
            for i in range(2):
                self.imageViews[i].set_from_pixbuf(self.clips[i][self.animFrame])
            self.animFrame = (self.animFrame + 1) % len(self.clips[0])
        GObject.timeout_add(50, self.animStep)

    def pickResult(self, result):
        if self.judgingClip:
            self.result = result
            self.judgingClip = False
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


win = PickerWindow()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()
