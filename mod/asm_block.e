int mod_identity (int n)
{
  return n;
}

asm (target = default) {
  (export "mod_identity" (func $add))
}
