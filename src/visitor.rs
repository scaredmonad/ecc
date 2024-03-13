use crate::parser::*;

// The visitor implementation we'll be using will allow multiple mutable references to a `Program`--no clone().
// - A first pass can enforce some known semantics and maybe collect local imports into a map for linking.
// - A separate pass for the type checker, after resolving imports.
// - A second pass will hold the printer for outputting to a `.wat` module.
// - For now, we'll use an asm block for imports.
pub(crate) trait ProgramVisitor {
    fn visit_program(&mut self, program: &mut Program) {
        for declaration in &mut program.declarations {
            self.visit_declaration(declaration);
        }
    }

    fn visit_declaration(&mut self, declaration: &mut Declaration) {
        match declaration {
            Declaration::Variable(var_decl) => self.visit_variable_declaration(var_decl),
            Declaration::Function(func_decl) => self.visit_function_declaration(func_decl),
            Declaration::AsmBlock(asm_block) => self.visit_asm_block(asm_block),
        }
    }

    fn visit_variable_declaration(&mut self, _var_decl: &mut VariableDeclaration) {}

    fn visit_function_declaration(&mut self, _func_decl: &mut FunctionDeclaration) {}

    fn visit_asm_block(&mut self, _asm_block: &mut AsmBlock) {}

    fn visit_expression(&mut self, expression: &mut Expression) {}

    // Should this be overriden bc it wont' prepend/append to the printer?
    fn visit_statement(&mut self, statement: &mut Statement) {
        match statement {
            Statement::ExpressionStatement(expr) => self.visit_expression(expr),
            Statement::VariableDeclaration(var_decl) => self.visit_variable_declaration(var_decl),
            Statement::Return(expr) => self.visit_expression(expr),
            _ => {}
        }
    }
}

impl Program {
    pub fn accept<V: ProgramVisitor>(&mut self, visitor: &mut V) {
        visitor.visit_program(self);
    }
}

impl VariableDeclaration {
    pub fn accept<V: ProgramVisitor>(&mut self, visitor: &mut V) {
        visitor.visit_variable_declaration(self);
    }
}

impl FunctionDeclaration {
    pub fn accept<V: ProgramVisitor>(&mut self, visitor: &mut V) {
        visitor.visit_function_declaration(self);

        // @fix: we now have to manually iterate via implementors. why?

        // for statement in &mut self.body {
        //     statement.accept(visitor);
        // }
    }
}

impl AsmBlock {
    pub fn accept<V: ProgramVisitor>(&mut self, visitor: &mut V) {
        visitor.visit_asm_block(self);
    }
}

impl Expression {
    pub fn accept<V: ProgramVisitor>(&mut self, visitor: &mut V) {
        visitor.visit_expression(self);
    }
}

impl Statement {
    pub fn accept<V: ProgramVisitor>(&mut self, visitor: &mut V) {
        visitor.visit_statement(self);
    }
}
