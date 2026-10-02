package a

type Widget struct {
	Count int
	Owner *Widget
}

func (w Widget) Render() int { return w.Count }

func useIt(w Widget) {
	a := widget.Make()
	show(widget.Limit)
	b := w.Count
	w.Render()
	c := w.Owner.Count
}
