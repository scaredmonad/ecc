(module
  (func $mod_add (param $a i32) (param $b i32) (result i32))
(i32.add (get_local $a) (get_local $b))
)
)