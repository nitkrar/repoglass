class Widget {
    var count = 0
    var owner: Widget? = null
    fun render() = this.count
    companion object {
        const val LIMIT = 3
        fun make(): Widget = Widget()
    }
}
fun useIt(w: Widget) {
    val a = Widget.make()
    show(Widget.LIMIT)
    val b = w.count
    w.render()
    val c = w.owner.count
}
