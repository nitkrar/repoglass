<?php
class Widget {
    const LIMIT = 3;
    public $count = 0;
    public static function make() { return new Widget(); }
    public function render() { return $this->count; }
}
function use_it(Widget $w) {
    $a = Widget::make();
    show(Widget::LIMIT);
    $b = $w->count;
    $w->render();
    $c = $w->owner->count;
}
