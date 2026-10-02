struct widget { int count; struct widget *owner; };
int use_it(struct widget w) {
    int b = w.count;
    int c = w.owner->count;
    return show(b + c);
}
