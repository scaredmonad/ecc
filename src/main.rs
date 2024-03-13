#![allow(dead_code)]
#![allow(unused_variables)]
// #![allow(unused_imports)]
use crate::lexer::collect_tokens;
use crate::parser::parse_program;
use crate::printer::WritePass;
use crate::semantics::SemanticPass;

mod lexer;
mod parser;
mod printer;
mod prog;
mod scope;
mod semantics;
mod visitor;

static STD_LINEAR_PRELUDE: &str = include_str!("prelude.wat");

pub(crate) fn compile(input: &str) -> Result<String, ()> {
    let mut tokens = collect_tokens(input);
    let mut program = parse_program(&mut tokens);
    let mut semantic_pass_visitor = SemanticPass::default();
    program.accept(&mut semantic_pass_visitor); // may fail, so sequential
    let mut write_pass_visitor = WritePass::default();
    program.accept(&mut write_pass_visitor);
    let output = write_pass_visitor.printer.to_string();
    Ok(output)
}

fn main() {
    use crate::prog::compile_from_env;
    compile_from_env();
}
