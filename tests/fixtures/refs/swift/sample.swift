struct Widget {
    static let LIMIT = 3
    var count = 0
    var owner: Widget? = nil
    static func make() -> Widget { Widget() }
    func render() {}
}
func useIt(w: Widget) {
    let a = Widget.make()
    show(Widget.LIMIT)
    let b = w.count
    w.render()
    let c = w.owner.count
}
