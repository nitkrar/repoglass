pub struct Widget { pub count: i32, pub owner: Box<Widget> }
impl Widget {
    pub const LIMIT: i32 = 3;
    pub fn make() -> Widget { todo!() }
    pub fn render(&self) -> i32 { self.count }
}
fn use_it(w: &Widget) {
    let a = Widget::make();
    show(Widget::LIMIT);
    let b = w.count;
    w.render();
    let c = w.owner.count;
}
