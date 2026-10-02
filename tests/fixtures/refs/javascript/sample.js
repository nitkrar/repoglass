class Widget {
  static LIMIT = 3;
  count = 0;
  static make() { return new Widget(); }
  render() { return this.count; }
}
function useIt(w) {
  const a = Widget.make();
  show(Widget.LIMIT);
  const b = w.count;
  w.render();
  const c = w.owner.count;
}
