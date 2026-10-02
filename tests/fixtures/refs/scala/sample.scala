class Widget {
  var count = 0
  var owner: Widget = null
  def render(): Int = this.count
}
object Widget {
  val LIMIT = 3
  def make(): Widget = new Widget()
}
object Use {
  def useIt(w: Widget): Unit = {
    val a = Widget.make()
    show(Widget.LIMIT)
    val b = w.count
    w.render()
    val c = w.owner.count
  }
}
