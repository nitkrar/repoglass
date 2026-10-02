class Widget {
    public const int LIMIT = 3;
    public int count;
    public Widget owner;
    public static Widget make() { return new Widget(); }
    public int render() { return this.count; }
}
class Use {
    void UseIt(Widget w) {
        var a = Widget.make();
        Show(Widget.LIMIT);
        var b = w.count;
        w.render();
        var c = w.owner.count;
    }
}
