const Widget = struct {
    count: i32,
    owner: *Widget,
    fn render(self: Widget) i32 { return self.count; }
};
fn useIt(w: Widget) void {
    const a = Widget.make();
    show(Widget.LIMIT);
    const b = w.count;
    w.render();
    const c = w.owner.count;
}
