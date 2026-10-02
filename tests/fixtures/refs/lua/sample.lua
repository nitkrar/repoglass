local Widget = {}
function use_it(w)
  local a = Widget.make()
  show(Widget.LIMIT)
  local b = w.count
  w:render()
  local c = w.owner.count
end
