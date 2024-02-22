#![allow(dead_code)]
#![allow(unused_variables)]
use std::str::Chars;

#[derive(Debug, Clone, PartialEq)]
enum TokenType {
    Import,
    Export,
    Extern,
    BoolLiteral(bool),
    Identifier(String),
    IntLiteral(i64),
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Comma,
    Semicolon,
    FieldAccessOp,
    Return,
    Plus,
    Minus,
    Equal,
    Mul,
    Div,
    Gt,
    Lt,
    EOF,
}

#[derive(Debug, Clone)]
struct Token {
    token_type: TokenType,
    lexeme: String,
}

struct Lexer<'a> {
    input: Chars<'a>,
    current_char: Option<char>,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        let mut chars = input.chars();
        let current_char = chars.next();
        Lexer {
            input: chars,
            current_char,
        }
    }

    fn advance(&mut self) {
        self.current_char = self.input.next();
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.current_char {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn consume_identifier(&mut self) -> String {
        let mut identifier = String::new();
        while let Some(c) = self.current_char {
            if c.is_alphanumeric() || c == '_' {
                identifier.push(c);
                self.advance();
            } else {
                break;
            }
        }
        identifier
    }

    fn consume_number(&mut self) -> i64 {
        let mut number = String::new();
        while let Some(c) = self.current_char {
            if c.is_digit(10) {
                number.push(c);
                self.advance();
            } else {
                break;
            }
        }
        number.parse().unwrap_or(0)
    }

    fn next_token(&mut self) -> Token {
        self.skip_whitespace();

        let token = match self.current_char {
            Some('(') => {
                self.advance();

                Token {
                    token_type: TokenType::LeftParen,
                    lexeme: "(".to_string(),
                }
            }

            Some(')') => {
                self.advance();

                Token {
                    token_type: TokenType::RightParen,
                    lexeme: ")".to_string(),
                }
            }

            Some('{') => {
                self.advance();

                Token {
                    token_type: TokenType::LeftBrace,
                    lexeme: "{".to_string(),
                }
            }

            Some('}') => {
                self.advance();

                Token {
                    token_type: TokenType::RightBrace,
                    lexeme: "}".to_string(),
                }
            }

            Some('[') => {
                self.advance();

                Token {
                    token_type: TokenType::LeftBracket,
                    lexeme: "[".to_string(),
                }
            }

            Some(']') => {
                self.advance();

                Token {
                    token_type: TokenType::RightBracket,
                    lexeme: "]".to_string(),
                }
            }

            Some(',') => {
                self.advance();

                Token {
                    token_type: TokenType::Comma,
                    lexeme: ",".to_string(),
                }
            }

            Some(';') => {
                self.advance();

                Token {
                    token_type: TokenType::Semicolon,
                    lexeme: ";".to_string(),
                }
            }

            Some('+') => {
                self.advance();

                Token {
                    token_type: TokenType::Plus,
                    lexeme: "+".to_string(),
                }
            }

            Some('-') => {
                self.advance();

                Token {
                    token_type: TokenType::Minus,
                    lexeme: "-".to_string(),
                }
            }

            Some('*') => {
                self.advance();

                Token {
                    token_type: TokenType::Mul,
                    lexeme: "*".to_string(),
                }
            }

            Some('/') => {
                self.advance();

                Token {
                    token_type: TokenType::Div,
                    lexeme: "/".to_string(),
                }
            }

            Some('>') => {
                self.advance();

                Token {
                    token_type: TokenType::Gt,
                    lexeme: ">".to_string(),
                }
            }

            Some('<') => {
                self.advance();

                Token {
                    token_type: TokenType::Lt,
                    lexeme: "<".to_string(),
                }
            }

            Some('=') => {
                self.advance();

                Token {
                    token_type: TokenType::Equal,
                    lexeme: "=".to_string(),
                }
            }

            Some('.') => {
                self.advance();

                Token {
                    token_type: TokenType::FieldAccessOp,
                    lexeme: ".".to_string(),
                }
            }

            Some('i') => {
                let identifier = self.consume_identifier();

                if identifier == "import" {
                    Token {
                        token_type: TokenType::Import,
                        lexeme: identifier,
                    }
                } else {
                    Token {
                        token_type: TokenType::Identifier(identifier.clone()),
                        lexeme: identifier,
                    }
                }
            }

            Some('r') => {
                let identifier = self.consume_identifier();

                if identifier == "return" {
                    Token {
                        token_type: TokenType::Return,
                        lexeme: identifier,
                    }
                } else {
                    Token {
                        token_type: TokenType::Identifier(identifier.clone()),
                        lexeme: identifier,
                    }
                }
            }

            Some('t') => {
                let identifier = self.consume_identifier();

                match identifier.as_str() {
                    "true" => Token {
                        token_type: TokenType::BoolLiteral(true),
                        lexeme: identifier,
                    },

                    _ => Token {
                        token_type: TokenType::Identifier(identifier.clone()),
                        lexeme: identifier,
                    },
                }
            }

            Some('f') => {
                let identifier = self.consume_identifier();

                match identifier.as_str() {
                    "false" => Token {
                        token_type: TokenType::BoolLiteral(false),
                        lexeme: identifier,
                    },

                    _ => Token {
                        token_type: TokenType::Identifier(identifier.clone()),
                        lexeme: identifier,
                    },
                }
            }

            Some('e') => {
                let identifier = self.consume_identifier();

                match identifier.as_str() {
                    "export" => Token {
                        token_type: TokenType::Export,
                        lexeme: identifier,
                    },

                    "extern" => Token {
                        token_type: TokenType::Extern,
                        lexeme: identifier,
                    },

                    _ => Token {
                        token_type: TokenType::Identifier(identifier.clone()),
                        lexeme: identifier,
                    },
                }
            }

            Some(c) if c.is_alphabetic() => {
                let identifier = self.consume_identifier();

                Token {
                    token_type: TokenType::Identifier(identifier.clone()),
                    lexeme: identifier,
                }
            }

            Some(c) if c.is_digit(10) => {
                let number = self.consume_number();

                Token {
                    token_type: TokenType::IntLiteral(number),
                    lexeme: number.to_string(),
                }
            }

            None => Token {
                token_type: TokenType::EOF,
                lexeme: "".to_string(),
            },

            _ => {
                panic!("Invalid character found: {:?}", self.current_char);
            }
        };

        token
    }
}

fn collect_tokens(input: &str) -> Vec<Token> {
    let mut lexer = Lexer::new(input);
    let mut tokens = Vec::new();

    loop {
        let token = lexer.next_token();
        tokens.push(token.clone());

        if token.token_type == TokenType::EOF {
            break;
        }
    }

    tokens
}

#[test]
fn assert_tokenize_input() {
    let input = r#"
        import std.fs;
        import std.fs.read;

        export int add(int a, int b) {
            return a + b;
        }

        int main() {
            int[] c = [
                add(1, 2),
                add(5, 10)
            ];
        }
    "#;
    let tokens = collect_tokens(input);
    assert_eq!(tokens.len() > 0, true);
}

#[derive(Debug, Clone)]
enum Type {
    Int,
    Bool,
}

impl From<&str> for Type {
    fn from(s: &str) -> Self {
        match s {
            "int" => Type::Int,
            "bool" => Type::Bool,
            _ => panic!("Invalid type"),
        }
    }
}

#[derive(Debug, Clone)]
struct Identifier(String);

#[derive(Debug, Clone)]
enum Expression {
    Uninit,
    IntLiteral(i64),
    BoolLiteral(bool),
    Variable(Identifier),
    Binary(Box<Expression>, BinaryOp, Box<Expression>),
    Comparison(Box<Expression>, CompareOp, Box<Expression>),
}

#[derive(Debug, Clone)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl From<&str> for BinaryOp {
    fn from(s: &str) -> Self {
        match s {
            "+" => BinaryOp::Add,
            "-" => BinaryOp::Sub,
            "*" => BinaryOp::Mul,
            "/" => BinaryOp::Div,
            _ => panic!("Invalid binary operator"),
        }
    }
}

#[derive(Debug, Clone)]
enum CompareOp {
    GreaterThan,
    LessThan,
    Equal,
}

impl From<&str> for CompareOp {
    fn from(s: &str) -> Self {
        match s {
            ">" => CompareOp::GreaterThan,
            "<" => CompareOp::LessThan,
            "==" => CompareOp::Equal,
            _ => panic!("Invalid comparison operator"),
        }
    }
}

#[derive(Debug, Clone)]
struct FunctionDeclaration {
    return_type: Type,
    identifier: Identifier,
    parameters: Vec<(Type, Identifier)>,
    body: Vec<Statement>,
}

#[derive(Debug, Clone)]
struct VariableDeclaration {
    data_type: Type,
    identifier: Identifier,
    value: Expression,
}

#[derive(Debug, Clone, PartialEq)]
enum Statement {
    VariableDeclaration(VariableDeclaration),
    Return(Expression),
    Expression(Expression),
}

#[derive(Debug, Clone)]
enum Declaration {
    Variable(VariableDeclaration),
    Function(FunctionDeclaration),
}

#[derive(Debug, Clone)]
struct Program {
    declarations: Vec<Declaration>,
}

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Type::Int, Type::Int) => true,
            (Type::Bool, Type::Bool) => true,
            _ => false,
        }
    }
}

impl PartialEq for Identifier {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialEq for BinaryOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BinaryOp::Add, BinaryOp::Add) => true,
            (BinaryOp::Sub, BinaryOp::Sub) => true,
            (BinaryOp::Mul, BinaryOp::Mul) => true,
            (BinaryOp::Div, BinaryOp::Div) => true,
            _ => false,
        }
    }
}

impl PartialEq for CompareOp {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CompareOp::GreaterThan, CompareOp::GreaterThan) => true,
            (CompareOp::LessThan, CompareOp::LessThan) => true,
            (CompareOp::Equal, CompareOp::Equal) => true,
            _ => false,
        }
    }
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Expression::IntLiteral(lhs), Expression::IntLiteral(rhs)) => lhs == rhs,
            (Expression::Variable(lhs), Expression::Variable(rhs)) => lhs == rhs,
            (Expression::Uninit, Expression::Uninit) => true,
            (Expression::BoolLiteral(true), Expression::BoolLiteral(true)) => true,
            (Expression::BoolLiteral(false), Expression::BoolLiteral(false)) => true,
            (
                Expression::Binary(lhs_a, lhs_op, lhs_b),
                Expression::Binary(rhs_a, rhs_op, rhs_b),
            ) => lhs_a == rhs_a && lhs_op == rhs_op && lhs_b == rhs_b,
            (
                Expression::Comparison(lhs_a, lhs_op, lhs_b),
                Expression::Comparison(rhs_a, rhs_op, rhs_b),
            ) => lhs_a == rhs_a && lhs_op == rhs_op && lhs_b == rhs_b,
            _ => false,
        }
    }
}

impl PartialEq for VariableDeclaration {
    fn eq(&self, other: &Self) -> bool {
        self.identifier == other.identifier && self.value == other.value
    }
}

impl PartialEq for FunctionDeclaration {
    fn eq(&self, other: &Self) -> bool {
        self.return_type == other.return_type
            && self.identifier == other.identifier
            && self.parameters == other.parameters
            && self.body == other.body
    }
}

impl PartialEq for Declaration {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Declaration::Variable(lhs), Declaration::Variable(rhs)) => lhs == rhs,
            (Declaration::Function(lhs), Declaration::Function(rhs)) => lhs == rhs,
            _ => false,
        }
    }
}

impl PartialEq for Program {
    fn eq(&self, other: &Self) -> bool {
        self.declarations == other.declarations
    }
}

fn parse_variable_declaration(tokens: &mut Vec<Token>) -> VariableDeclaration {
    // Minimum tokens (should proly be 5 for initialized).
    if tokens.len() < 4 {
        panic!("Invalid declaration syntax: Insufficient tokens");
    }

    // Eat LHS type identifier.
    let type_identifier = match tokens.remove(0) {
        Token {
            token_type: TokenType::Identifier(_),
            lexeme,
        } => lexeme,
        _ => panic!("Invalid declaration syntax: Type identifier expected"),
    };

    // Eat LHS variable ident.
    let variable_identifier = match tokens.remove(0) {
        Token {
            token_type: TokenType::Identifier(_),
            lexeme,
        } => lexeme,
        _ => panic!("Invalid declaration syntax: Variable identifier expected"),
    };

    // Check if there's an initialization expression.
    let value = if matches!(
        tokens.get(0),
        Some(&Token {
            token_type: TokenType::Equal,
            ..
        })
    ) {
        tokens.remove(0); // Eat '=' token
        parse_expression(tokens)
    } else {
        Expression::Uninit // Variable is uninitialized.
    };

    VariableDeclaration {
        data_type: Type::from(type_identifier.as_str()),
        identifier: Identifier(variable_identifier),
        value,
    }
}

#[test]
fn assert_parse_var_decl() {
    let input = "int a = 5;";
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Variable(VariableDeclaration {
                data_type: Type::Int,
                identifier: Identifier("a".into()),
                value: Expression::IntLiteral(5)
            })]
        }
    );
}

#[test]
fn assert_parse_multi_var_decl() {
    let input = "int a = 5; int b = 9;";
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int,
                    identifier: Identifier("a".into()),
                    value: Expression::IntLiteral(5)
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int,
                    identifier: Identifier("b".into()),
                    value: Expression::IntLiteral(9)
                })
            ]
        }
    );
}

#[test]
fn assert_parse_uninit_var_decl() {
    let input = "int a;";
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Variable(VariableDeclaration {
                data_type: Type::Int,
                identifier: Identifier("a".into()),
                value: Expression::Uninit
            })]
        }
    );
}

fn parse_expression(tokens: &mut Vec<Token>) -> Expression {
    // Parse the first operand of the expression.
    let mut left_operand = match tokens.remove(0).token_type {
        TokenType::BoolLiteral(value) => Expression::BoolLiteral(value),
        TokenType::IntLiteral(value) => Expression::IntLiteral(value),
        TokenType::Identifier(name) => Expression::Variable(Identifier(name)),
        _ => panic!("Invalid expression"),
    };

    // Check if there's a binary operator.
    while let Some(operator_token) = tokens.get(0).cloned() {
        match operator_token.token_type {
            TokenType::Plus | TokenType::Minus | TokenType::Mul | TokenType::Div => {
                tokens.remove(0); // Eat op token.
                let right_operand = parse_expression(tokens);
                left_operand = Expression::Binary(
                    Box::new(left_operand),
                    BinaryOp::from(operator_token.lexeme.as_str()),
                    Box::new(right_operand),
                );
            }

            TokenType::Gt | TokenType::Lt | TokenType::Equal => {
                tokens.remove(0); // Eat op token.
                let right_operand = parse_expression(tokens);
                left_operand = Expression::Comparison(
                    Box::new(left_operand),
                    CompareOp::from(operator_token.lexeme.as_str()),
                    Box::new(right_operand),
                );
            }
            _ => break,
        }
    }

    left_operand
}

#[test]
fn assert_parse_var_decl_bool_literal() {
    let input = "bool T = true; bool F = false;";
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int,
                    identifier: Identifier("T".into()),
                    value: Expression::BoolLiteral(true)
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int,
                    identifier: Identifier("F".into()),
                    value: Expression::BoolLiteral(false)
                })
            ]
        }
    );
}

#[test]
fn assert_parse_var_decl_binary_expr() {
    let input = r#"
        int a = 5 + 9;
        int b = 8 - 6;
        int c = 6 * 2;
        int d = 1 / 10;
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int,
                    identifier: Identifier("a".into()),
                    value: Expression::Binary(
                        Box::new(Expression::IntLiteral(5)),
                        BinaryOp::Add,
                        Box::new(Expression::IntLiteral(9))
                    )
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int,
                    identifier: Identifier("b".into()),
                    value: Expression::Binary(
                        Box::new(Expression::IntLiteral(8)),
                        BinaryOp::Sub,
                        Box::new(Expression::IntLiteral(6))
                    )
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int,
                    identifier: Identifier("c".into()),
                    value: Expression::Binary(
                        Box::new(Expression::IntLiteral(6)),
                        BinaryOp::Mul,
                        Box::new(Expression::IntLiteral(2))
                    )
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int,
                    identifier: Identifier("d".into()),
                    value: Expression::Binary(
                        Box::new(Expression::IntLiteral(1)),
                        BinaryOp::Div,
                        Box::new(Expression::IntLiteral(10))
                    )
                }),
            ]
        }
    );
}

#[test]
fn assert_parse_var_decl_compare_expr() {
    let input = "bool T = 7 > 2;";
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Variable(VariableDeclaration {
                data_type: Type::Int,
                identifier: Identifier("T".into()),
                value: Expression::Comparison(
                    Box::new(Expression::IntLiteral(7)),
                    CompareOp::GreaterThan,
                    Box::new(Expression::IntLiteral(2))
                )
            }),]
        }
    );
}

fn parse_function_declaration(tokens: &mut Vec<Token>) -> Result<FunctionDeclaration, String> {
    let return_type = match tokens.remove(0).token_type {
        TokenType::Identifier(ref name) => Type::from(name.as_str()),
        _ => panic!("Invalid return type"),
    };

    let identifier = match tokens.remove(0).token_type {
        TokenType::Identifier(ref name) => Identifier(name.clone()),
        _ => panic!("Invalid function identifier"),
    };

    let mut parameters = Vec::new();
    match tokens.remove(0).token_type {
        TokenType::LeftParen => (),
        _ => panic!("Expected '(' after function identifier"),
    }

    while tokens[0].token_type != TokenType::RightParen {
        let param_type = match tokens.remove(0).token_type {
            TokenType::Identifier(ref name) => Type::from(name.as_str()),
            _ => panic!("Invalid parameter type"),
        };

        let param_name = match tokens.remove(0).token_type {
            TokenType::Identifier(ref name) => Identifier(name.clone()),
            _ => panic!("Invalid parameter name"),
        };

        parameters.push((param_type, param_name));

        if tokens[0].token_type == TokenType::Comma {
            tokens.remove(0);
        }
    }

    tokens.remove(0); // Eat ')'

    let mut body = Vec::new();

    match tokens.remove(0).token_type {
        TokenType::LeftBrace => (),
        _ => panic!("Expected '{{' at the beginning of function body"),
    }

    while tokens[0].token_type != TokenType::RightBrace {
        match tokens[0].token_type {
            TokenType::Identifier(_) => {
                let variable_declaration = parse_variable_declaration(tokens);
                body.push(Statement::VariableDeclaration(variable_declaration));
                if tokens[0].token_type == TokenType::Semicolon {
                    tokens.remove(0); // Eat ';'
                }
            }

            TokenType::Return => {
                tokens.remove(0); // Eat 'return'
                let expression = parse_expression(tokens);
                body.push(Statement::Return(expression));
                if tokens[0].token_type == TokenType::Semicolon {
                    tokens.remove(0); // Eat ';'
                }
            }

            TokenType::EOF | TokenType::Semicolon => break, // /!\
            _ => panic!("Unexpected token in function body"),
        }
    }

    tokens.remove(0); // Eat '}'

    Ok(FunctionDeclaration {
        return_type,
        identifier,
        parameters,
        body,
    })
}

#[test]
fn assert_parse_fn_decl_empty() {
    let input = r#"
        int f() {}
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int,
                identifier: Identifier("f".into()),
                parameters: vec![],
                body: vec![]
            }),]
        }
    );
}

#[test]
fn assert_parse_fn_decl() {
    let input = r#"
        int f(int a, bool b, int c, bool d) {
            int k = 7 + 10;
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int,
                identifier: Identifier("f".into()),
                parameters: vec![
                    (Type::Int, Identifier("a".into())),
                    (Type::Bool, Identifier("b".into())),
                    (Type::Int, Identifier("c".into())),
                    (Type::Bool, Identifier("d".into())),
                ],
                body: vec![Statement::VariableDeclaration(VariableDeclaration {
                    data_type: Type::Int,
                    identifier: Identifier("k".into()),
                    value: Expression::Binary(
                        Box::new(Expression::IntLiteral(7)),
                        BinaryOp::Add,
                        Box::new(Expression::IntLiteral(10))
                    )
                })]
            }),]
        }
    );
}

#[test]
fn assert_parse_fn_decl_returns() {
    let input = r#"
        bool gt(int a, int b) {
            return 8 > 2;
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Bool,
                identifier: Identifier("gt".into()),
                parameters: vec![
                    (Type::Int, Identifier("a".into())),
                    (Type::Int, Identifier("b".into())),
                ],
                body: vec![Statement::Return(Expression::Comparison(
                    Box::new(Expression::IntLiteral(8)),
                    CompareOp::GreaterThan,
                    Box::new(Expression::IntLiteral(2))
                ))]
            }),]
        }
    );
}

#[test]
fn assert_parse_fn_decl_order() {
    let input = r#"
        int i = 0;
        int i(int j) {
            int i;
        }
        int j = 0;
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int,
                    identifier: Identifier("i".into()),
                    value: Expression::IntLiteral(0)
                }),
                Declaration::Function(FunctionDeclaration {
                    return_type: Type::Int,
                    identifier: Identifier("i".into()),
                    parameters: vec![(Type::Int, Identifier("j".into())),],
                    body: vec![Statement::VariableDeclaration(VariableDeclaration {
                        data_type: Type::Int,
                        identifier: Identifier("i".into()),
                        value: Expression::Uninit
                    })]
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int,
                    identifier: Identifier("j".into()),
                    value: Expression::IntLiteral(0)
                }),
            ]
        }
    );
}

fn parse_program(tokens: &mut Vec<Token>) -> Program {
    let mut declarations = Vec::new();

    while !tokens.is_empty() {
        match tokens[0].token_type {
            TokenType::Identifier(_) => {
                // Check if it's a fn or var decl since our syntax for both is similar.
                let is_function = tokens
                    .iter()
                    .take_while(|t| {
                        t.token_type != TokenType::LeftBrace && t.token_type != TokenType::Semicolon
                    })
                    .any(|t| matches!(t.token_type, TokenType::LeftParen));

                if is_function {
                    if let Ok(function_declaration) = parse_function_declaration(tokens) {
                        declarations.push(Declaration::Function(function_declaration));
                    } else {
                        panic!("Failed to parse function declaration");
                    }
                } else {
                    let variable_declaration = parse_variable_declaration(tokens);

                    declarations.push(Declaration::Variable(variable_declaration));

                    if tokens[0].token_type == TokenType::Semicolon {
                        tokens.remove(0); // Eat ';'
                    }
                }
            }
            _ => {
                tokens.remove(0); // /!\ skip unexpected tokens.
            }
        }
    }

    Program { declarations }
}

fn main() {
    let input = r#"
        int mul(int a, int b) {
            int k = a * b;

            return k;
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    dbg!(program);
}
