class Widget {
    static final int LIMIT = 3;
    int count;
    Widget owner;
    static Widget make() { return new Widget(); }
    int render() { return this.count; }
}
class Use {
    void useIt(Widget w) {
        Widget a = Widget.make();
        show(Widget.LIMIT);
        int b = w.count;
        w.render();
        int c = w.owner.count;
    }
}
