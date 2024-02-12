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
struct VariableDeclaration {
    data_type: Type,
    identifier: Identifier,
    value: Expression,
}

#[derive(Debug, Clone)]
enum Declaration {
    Variable(VariableDeclaration),
}

#[derive(Debug, Clone)]
struct Program {
    declarations: Vec<Declaration>,
}

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Type::Int, Type::Int) => true,
            // _ => false,
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

impl PartialEq for Declaration {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Declaration::Variable(lhs), Declaration::Variable(rhs)) => lhs == rhs,
            // _ => false,
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
        data_type: Type::Int, // @todo: derive types with From<String>.
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

fn parse_program(tokens: &mut Vec<Token>) -> Program {
    let mut declarations = Vec::new();

    while !tokens.is_empty() {
        match tokens[0].token_type {
            TokenType::Identifier(_) => {
                let declaration = parse_variable_declaration(tokens);
                declarations.push(Declaration::Variable(declaration));
            }
            _ => {
                tokens.remove(0); // /!\
            }
        }
    }

    Program { declarations }
}

fn main() {
    let input = r#"
        bool b = 7 > 2;
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    dbg!(program);
}
