class Widget:
    LIMIT = 3

    def make():
        return Widget()

    def render(self):
        return self.count


def use_it(w: Widget):
    a = Widget.make()
    show(Widget.LIMIT)
    b = w.count
    w.render()
    c = w.owner.count
