; Reference captures for Rust, beyond rust-tags.scm's bare and method
; calls, macros and impls.

; A path's last segment, called or not: `open` in `crypto::open(..)`,
; `LIMIT` in `Widget::LIMIT`. A path call is the ordinary form for
; anything not imported bare.
(scoped_identifier
  name: (identifier) @name.reference.member) @reference.member

; The segment before `::`, always a type or a module: `Widget` in
; `Widget::make()`.
(scoped_identifier
  path: (identifier) @name.reference.type) @reference.type

; A field, called or read: `count` in `w.count`.
(field_expression
  field: (field_identifier) @name.reference.member) @reference.member
