#![allow(dead_code)]
#![allow(unused_variables)]

#[derive(Debug, Clone, PartialEq)]
enum TokenType {
    Asm,
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
    Dot,
    If,
    Else,
    For,
    While,
    Return,
    Equal,
    EqEq,
    Plus,
    Minus,
    Mul,
    Div,
    Bang,
    PlusEqual,
    MinusEqual,
    MulEqual,
    DivEqual,
    BangEqual,
    Gt,
    Lt,
    GtEq,
    LtEq,
    Dollar,
    DoubleQuote,
    EOF,
}

#[derive(Debug, Clone)]
struct Token {
    token_type: TokenType,
    lexeme: String,
}

struct Lexer<'a> {
    input: std::str::Chars<'a>,
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

            Some('$') => {
                self.advance();

                Token {
                    token_type: TokenType::Dollar,
                    lexeme: "$".to_string(),
                }
            }

            Some('"') => {
                self.advance();

                Token {
                    token_type: TokenType::DoubleQuote,
                    lexeme: "\"".to_string(),
                }
            }

            Some('+') => {
                self.advance();

                if self.current_char == Some('=') {
                    self.advance();

                    Token {
                        token_type: TokenType::PlusEqual,
                        lexeme: "+=".to_string(),
                    }
                } else {
                    Token {
                        token_type: TokenType::Plus,
                        lexeme: "+".to_string(),
                    }
                }
            }

            Some('-') => {
                self.advance();

                if self.current_char == Some('=') {
                    self.advance();

                    Token {
                        token_type: TokenType::MinusEqual,
                        lexeme: "-=".to_string(),
                    }
                } else {
                    Token {
                        token_type: TokenType::Minus,
                        lexeme: "-".to_string(),
                    }
                }
            }

            Some('*') => {
                self.advance();

                if self.current_char == Some('=') {
                    self.advance();

                    Token {
                        token_type: TokenType::MulEqual,
                        lexeme: "*=".to_string(),
                    }
                } else {
                    Token {
                        token_type: TokenType::Mul,
                        lexeme: "*".to_string(),
                    }
                }
            }

            Some('/') => {
                self.advance();

                if self.current_char == Some('=') {
                    self.advance();

                    Token {
                        token_type: TokenType::DivEqual,
                        lexeme: "/=".to_string(),
                    }
                } else {
                    Token {
                        token_type: TokenType::Div,
                        lexeme: "/".to_string(),
                    }
                }
            }

            Some('!') => {
                self.advance();

                if self.current_char == Some('=') {
                    self.advance();

                    Token {
                        token_type: TokenType::BangEqual,
                        lexeme: "!=".to_string(),
                    }
                } else {
                    Token {
                        token_type: TokenType::Bang,
                        lexeme: "!".to_string(),
                    }
                }
            }

            Some('>') => {
                self.advance();

                if self.current_char == Some('=') {
                    self.advance();

                    Token {
                        token_type: TokenType::GtEq,
                        lexeme: ">=".to_string(),
                    }
                } else {
                    Token {
                        token_type: TokenType::Gt,
                        lexeme: ">".to_string(),
                    }
                }
            }

            Some('<') => {
                self.advance();

                if self.current_char == Some('=') {
                    self.advance();

                    Token {
                        token_type: TokenType::LtEq,
                        lexeme: "<=".to_string(),
                    }
                } else {
                    Token {
                        token_type: TokenType::Lt,
                        lexeme: "<".to_string(),
                    }
                }
            }

            Some('=') => {
                self.advance();

                if self.current_char == Some('=') {
                    self.advance();

                    Token {
                        token_type: TokenType::EqEq,
                        lexeme: "==".to_string(),
                    }
                } else {
                    Token {
                        token_type: TokenType::Equal,
                        lexeme: "=".to_string(),
                    }
                }
            }

            // Some('=') => {
            //     self.advance();

            //     Token {
            //         token_type: TokenType::Equal,
            //         lexeme: "=".to_string(),
            //     }
            // }
            Some('.') => {
                self.advance();

                Token {
                    token_type: TokenType::Dot,
                    lexeme: ".".to_string(),
                }
            }

            Some('i') => {
                let identifier = self.consume_identifier();

                match identifier.as_str() {
                    "import" => Token {
                        token_type: TokenType::Import,
                        lexeme: identifier,
                    },

                    "if" => Token {
                        token_type: TokenType::If,
                        lexeme: identifier,
                    },

                    _ => Token {
                        token_type: TokenType::Identifier(identifier.clone()),
                        lexeme: identifier,
                    },
                }
            }

            Some('a') => {
                let identifier = self.consume_identifier();

                if identifier == "asm" {
                    Token {
                        token_type: TokenType::Asm,
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

                    "for" => Token {
                        token_type: TokenType::For,
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
                    "else" => Token {
                        token_type: TokenType::Else,
                        lexeme: identifier,
                    },

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

            Some('w') => {
                let identifier = self.consume_identifier();

                if identifier == "while" {
                    Token {
                        token_type: TokenType::While,
                        lexeme: identifier,
                    }
                } else {
                    Token {
                        token_type: TokenType::Identifier(identifier.clone()),
                        lexeme: identifier,
                    }
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
        <= >= == ! != ! =
    "#;
    let tokens = collect_tokens(input);
    dbg!(tokens.clone());
    assert_eq!(tokens.len() > 0, true);
}

// Ref: https://webassembly.github.io/spec/core/syntax/types.html
#[derive(Debug, Clone)]
enum Type {
    Int32,
    Int64,
    Uint32,
    Uint64,
    Bool,
    Char, // @todo: funcref & externref which work with the output.
          /* Entries for SIMD vector types, which we can represent as packed floats or
          4 blocks of 32-bit ints, etc, accordingly. */
}

impl From<&str> for Type {
    fn from(s: &str) -> Self {
        match s {
            "i32" => Type::Int32,
            "i64" => Type::Int64,
            "u32" => Type::Uint32,
            "u64" => Type::Uint64,
            "bool" => Type::Bool,
            "char" => Type::Char,
            _ => panic!("Invalid type"),
        }
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let type_str = match self {
            Type::Int32 => "i32",
            Type::Int64 => "i64",
            Type::Uint32 => "u32",
            Type::Uint64 => "u64",
            Type::Bool => "bool",
            Type::Char => "char",
        };
        write!(f, "{}", type_str)
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
    Assignment(Box<AssignmentExpression>),
    Call(Box<CallExpression>),
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
    GreaterThan,        // >
    LessThan,           // <
    GreaterThanOrEqual, // >=
    LessThanOrEqual,    // <=
    StrictEqual,        // ==
    StrictUnequal,      // !=
}

impl From<&str> for CompareOp {
    fn from(s: &str) -> Self {
        match s {
            ">" => CompareOp::GreaterThan,
            "<" => CompareOp::LessThan,
            ">=" => CompareOp::GreaterThanOrEqual,
            "<=" => CompareOp::LessThanOrEqual,
            "==" => CompareOp::StrictEqual,
            "!=" => CompareOp::StrictUnequal,
            _ => panic!("Invalid comparison operator"),
        }
    }
}

#[derive(Debug, Clone)]
struct AssignmentExpression {
    left: Identifier,
    operator: AssignmentOperator,
    right: Box<Expression>,
}

#[derive(Debug, Clone)]
enum AssignmentOperator {
    Assign,    // '='
    AddAssign, // '+='
    SubAssign, // '-='
    MulAssign, // '*='
    DivAssign, // '/='
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
struct IfStatement {
    condition: Box<Expression>,
    body: Vec<Statement>,
    alternative: Option<Vec<Statement>>,
}

#[derive(Debug, Clone, PartialEq)]
struct WhileStatement {
    condition: Box<Expression>,
    body: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
struct ForLoop {
    init: VariableDeclaration,
    test: Expression,
    update: Expression,
    body: Option<Vec<Statement>>,
}

#[derive(Debug, Clone)]
struct CallExpression {
    callee: Identifier,
    type_parameters: Option<Vec<Type>>,
    parameters: Vec<Expression>,
}

#[derive(Debug, Clone)]
struct AsmBlock {
    target: Expression,
    instr_field: String,
}

#[derive(Debug, Clone, PartialEq)]
enum Statement {
    VariableDeclaration(VariableDeclaration),
    Return(Expression),
    Expression(Expression),
    IfStatement(IfStatement),
    WhileStatement(WhileStatement),
    ExpressionStatement(Expression),
    ForLoop(ForLoop),
}

#[derive(Debug, Clone)]
enum Declaration {
    Variable(VariableDeclaration),
    Function(FunctionDeclaration),
    AsmBlock(AsmBlock),
}

#[derive(Debug, Clone)]
struct Program {
    declarations: Vec<Declaration>,
}

impl PartialEq for Type {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Type::Int32, Type::Int32) => true,
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
            (CompareOp::GreaterThanOrEqual, CompareOp::GreaterThanOrEqual) => true,
            (CompareOp::LessThanOrEqual, CompareOp::LessThanOrEqual) => true,
            (CompareOp::StrictEqual, CompareOp::StrictEqual) => true,
            (CompareOp::StrictUnequal, CompareOp::StrictUnequal) => true,
            _ => false,
        }
    }
}

impl PartialEq for AssignmentOperator {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AssignmentOperator::Assign, AssignmentOperator::Assign) => true,
            (AssignmentOperator::AddAssign, AssignmentOperator::AddAssign) => true,
            (AssignmentOperator::SubAssign, AssignmentOperator::SubAssign) => true,
            (AssignmentOperator::MulAssign, AssignmentOperator::MulAssign) => true,
            (AssignmentOperator::DivAssign, AssignmentOperator::DivAssign) => true,
            _ => false,
        }
    }
}

impl PartialEq for AssignmentExpression {
    fn eq(&self, other: &Self) -> bool {
        self.left == other.left && self.operator == other.operator && self.right == other.right
    }
}

impl PartialEq for CallExpression {
    fn eq(&self, other: &Self) -> bool {
        self.callee == other.callee
            && self.type_parameters == other.type_parameters
            && self.parameters == other.parameters
    }
}

impl PartialEq for AsmBlock {
    fn eq(&self, other: &Self) -> bool {
        self.target == other.target && self.instr_field == other.instr_field
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
            (Expression::Assignment(ass_expr), Expression::Assignment(other_ass_expr)) => {
                ass_expr == other_ass_expr
            }
            (Expression::Call(call_expr), Expression::Call(other_call_expr)) => {
                call_expr == other_call_expr
            }
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
            (Declaration::AsmBlock(lhs), Declaration::AsmBlock(rhs)) => lhs == rhs,
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
    let input = "i32 a = 5;";
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Variable(VariableDeclaration {
                data_type: Type::Int32,
                identifier: Identifier("a".into()),
                value: Expression::IntLiteral(5)
            })]
        }
    );
}

#[test]
fn assert_parse_multi_var_decl() {
    let input = "i32 a = 5; i32 b = 9;";
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("a".into()),
                    value: Expression::IntLiteral(5)
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("b".into()),
                    value: Expression::IntLiteral(9)
                })
            ]
        }
    );
}

#[test]
fn assert_parse_uninit_var_decl() {
    let input = "i32 a;";
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Variable(VariableDeclaration {
                data_type: Type::Int32,
                identifier: Identifier("a".into()),
                value: Expression::Uninit
            })]
        }
    );
}

fn parse_assignment_expression(tokens: &mut Vec<Token>) -> Result<Expression, String> {
    let left = match tokens.remove(0).token_type {
        TokenType::Identifier(name) => Identifier(name),
        _ => return Err("Expected identifier on the left side of assignment".to_string()),
    };

    let operator = match tokens.remove(0).token_type {
        TokenType::Equal => AssignmentOperator::Assign,
        TokenType::PlusEqual => AssignmentOperator::AddAssign,
        TokenType::MinusEqual => AssignmentOperator::SubAssign,
        TokenType::MulEqual => AssignmentOperator::MulAssign,
        TokenType::DivEqual => AssignmentOperator::DivAssign,
        _ => return Err("Expected assignment operator".to_string()),
    };

    let right = parse_expression(tokens);

    Ok(Expression::Assignment(Box::new(AssignmentExpression {
        left,
        operator,
        right: Box::new(right),
    })))
}

#[test]
fn assert_parse_compound_assign_expr() {
    let input = r#"
        i32 f() {
            k = 2;
            k += 1;
            k -= 1;
            k *= 1;
            k /= 1;
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int32,
                identifier: Identifier("f".into()),
                parameters: vec![],
                body: vec![
                    Statement::ExpressionStatement(Expression::Assignment(Box::new(
                        AssignmentExpression {
                            left: Identifier("k".into()),
                            operator: AssignmentOperator::Assign,
                            right: Box::new(Expression::IntLiteral(2))
                        }
                    ))),
                    Statement::ExpressionStatement(Expression::Assignment(Box::new(
                        AssignmentExpression {
                            left: Identifier("k".into()),
                            operator: AssignmentOperator::AddAssign,
                            right: Box::new(Expression::IntLiteral(1))
                        }
                    ))),
                    Statement::ExpressionStatement(Expression::Assignment(Box::new(
                        AssignmentExpression {
                            left: Identifier("k".into()),
                            operator: AssignmentOperator::SubAssign,
                            right: Box::new(Expression::IntLiteral(1))
                        }
                    ))),
                    Statement::ExpressionStatement(Expression::Assignment(Box::new(
                        AssignmentExpression {
                            left: Identifier("k".into()),
                            operator: AssignmentOperator::MulAssign,
                            right: Box::new(Expression::IntLiteral(1))
                        }
                    ))),
                    Statement::ExpressionStatement(Expression::Assignment(Box::new(
                        AssignmentExpression {
                            left: Identifier("k".into()),
                            operator: AssignmentOperator::DivAssign,
                            right: Box::new(Expression::IntLiteral(1))
                        }
                    ))),
                ]
            }),]
        }
    );
}

fn parse_call_expression(tokens: &mut Vec<Token>) -> Result<Expression, String> {
    let callee = match tokens.remove(0).token_type {
        TokenType::Identifier(name) => Identifier(name),
        _ => return Err("Expected function name".to_string()),
    };

    let type_parameters = if matches!(tokens.get(0).map(|t| &t.token_type), Some(TokenType::Lt)) {
        tokens.remove(0); // Eat '<'
        let mut types = Vec::new();
        while !matches!(tokens.get(0).map(|t| &t.token_type), Some(TokenType::Gt)) {
            match &tokens.get(0) {
                Some(token) => match &token.token_type {
                    TokenType::Identifier(type_str) => {
                        types.push(Type::from(type_str.as_str()));
                        tokens.remove(0); // Eat type ident
                    }
                    _ => return Err("Expected type identifier within type parameters".to_string()),
                },
                None => {
                    return Err("Unexpected end of tokens while parsing type parameters".to_string())
                }
            }

            if matches!(tokens.get(0).map(|t| &t.token_type), Some(TokenType::Comma)) {
                tokens.remove(0); // Eat ',' allows parsing next type ident
            }
        }
        tokens.remove(0); // Eat '>'
        Some(types)
    } else {
        None
    };

    if tokens.remove(0).token_type != TokenType::LeftParen {
        return Err("Expected '(' after function name".to_string()); // Eat '('
    }

    let mut parameters = Vec::new();
    while !matches!(
        tokens.get(0).map(|t| &t.token_type),
        Some(TokenType::RightParen)
    ) {
        let parameter = parse_expression(tokens);
        parameters.push(parameter);

        if matches!(tokens.get(0).map(|t| &t.token_type), Some(TokenType::Comma)) {
            tokens.remove(0); // Eat ',' allows parsing the next call parameter
        }
    }
    tokens.remove(0); // Eat ')'

    Ok(Expression::Call(Box::new(CallExpression {
        callee,
        type_parameters,
        parameters,
    })))
}

#[test]
fn assert_parse_call_expr_stmt() {
    let input = r#"
        i32 f() {
            load<i32>(2, 3);
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int32,
                identifier: Identifier("f".into()),
                parameters: vec![],
                body: vec![Statement::ExpressionStatement(Expression::Call(Box::new(
                    CallExpression {
                        callee: Identifier("load".into()),
                        type_parameters: Some(vec![Type::Int32]),
                        parameters: vec![Expression::IntLiteral(2), Expression::IntLiteral(3),]
                    }
                )))]
            }),]
        }
    );
}

#[test]
fn assert_parse_call_expr_var_decl() {
    let input = r#"
        i32 f() {
            i32 a = load<i32, bool>(2, 3, k > 7);
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int32,
                identifier: Identifier("f".into()),
                parameters: vec![],
                body: vec![Statement::VariableDeclaration(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("a".into()),
                    value: Expression::Call(Box::new(CallExpression {
                        callee: Identifier("load".into()),
                        type_parameters: Some(vec![Type::Int32, Type::Bool]),
                        parameters: vec![
                            Expression::IntLiteral(2),
                            Expression::IntLiteral(3),
                            Expression::Comparison(
                                Box::new(Expression::Variable(Identifier("k".into()))),
                                CompareOp::GreaterThan,
                                Box::new(Expression::IntLiteral(7))
                            )
                        ]
                    }))
                })]
            }),]
        }
    );
}

fn parse_asm_block(tokens: &mut Vec<Token>) -> Result<AsmBlock, String> {
    match tokens.remove(0).token_type {
        TokenType::Asm => (), // Expect and eat `asm`
        _ => return Err("Expected 'asm' keyword".to_string()),
    }

    if tokens.remove(0).token_type != TokenType::LeftParen {
        return Err("Expected '(' after 'asm' keyword".to_string());
    }

    let target = parse_assignment_expression(tokens)?;

    if tokens.remove(0).token_type != TokenType::RightParen {
        return Err("Expected ')' after target specification".to_string());
    }

    if tokens.remove(0).token_type != TokenType::LeftBrace {
        return Err("Expected '{' to start the asm block".to_string()); // Eat '{'
    }

    // Capture everything inside the braces as a raw string.
    let mut instr_field = String::new();
    while let Some(token) = tokens.get(0) {
        match token.token_type {
            TokenType::RightBrace => {
                tokens.remove(0); // Eat '}'
                break;
            }
            _ => {
                // Add the current token's lexeme to instr_field and eat.
                instr_field.push_str(&token.lexeme);
                tokens.remove(0);
            }
        }
    }

    Ok(AsmBlock {
        target,
        instr_field,
    })
}

#[test]
fn assert_parse_asm_block_empty() {
    let input = r#"
        i32 a = 5;
        asm (target = default) {}
        i32 f() {}
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("a".into()),
                    value: Expression::IntLiteral(5)
                }),
                Declaration::AsmBlock(AsmBlock {
                    target: Expression::Assignment(Box::new(AssignmentExpression {
                        left: Identifier("target".into()),
                        operator: AssignmentOperator::Assign,
                        right: Box::new(Expression::Variable(Identifier("default".into())))
                    })),
                    instr_field: "".into()
                }),
                Declaration::Function(FunctionDeclaration {
                    return_type: Type::Int32,
                    identifier: Identifier("f".into()),
                    parameters: vec![],
                    body: vec![]
                }),
            ]
        }
    );
}

fn parse_expression(tokens: &mut Vec<Token>) -> Expression {
    // Parse the first operand of the expression.
    let mut left_operand = match tokens.remove(0).token_type {
        TokenType::BoolLiteral(value) => Expression::BoolLiteral(value),
        TokenType::IntLiteral(value) => Expression::IntLiteral(value),
        // TokenType::Identifier(name) => Expression::Variable(Identifier(name)),
        TokenType::Identifier(name) => {
            // We're checking if the ident is followed by a call expression start, so either:
            // 1. LeftParen for the expr itself (no tparams)
            // 2. Lt for type parameters
            match tokens.get(0).map(|t| &t.token_type) {
                Some(TokenType::LeftParen) | Some(TokenType::Lt) => {
                    // Put back the identifier token and call parse_call_expression
                    // This behavior (and eat, eat_while) must be abstracted away.
                    tokens.insert(
                        0,
                        Token {
                            token_type: TokenType::Identifier(name.clone()),
                            lexeme: name.clone(),
                        },
                    );

                    parse_call_expression(tokens).unwrap()
                }
                _ => Expression::Variable(Identifier(name)), // Regular ident.
            }
        }
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

            TokenType::Gt
            | TokenType::Lt
            | TokenType::Equal
            | TokenType::GtEq
            | TokenType::LtEq
            | TokenType::EqEq
            | TokenType::BangEqual => {
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
                    data_type: Type::Int32,
                    identifier: Identifier("T".into()),
                    value: Expression::BoolLiteral(true)
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
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
        i32 a = 5 + 9;
        i32 b = 8 - 6;
        i32 c = 6 * 2;
        i32 d = 1 / 10;
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("a".into()),
                    value: Expression::Binary(
                        Box::new(Expression::IntLiteral(5)),
                        BinaryOp::Add,
                        Box::new(Expression::IntLiteral(9))
                    )
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("b".into()),
                    value: Expression::Binary(
                        Box::new(Expression::IntLiteral(8)),
                        BinaryOp::Sub,
                        Box::new(Expression::IntLiteral(6))
                    )
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("c".into()),
                    value: Expression::Binary(
                        Box::new(Expression::IntLiteral(6)),
                        BinaryOp::Mul,
                        Box::new(Expression::IntLiteral(2))
                    )
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
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
    let input = r#"
        bool T = 7 > 2;
        bool T = 7 < 2;
        bool T = 7 >= 2;
        bool T = 7 <= 2;
        bool T = 7 == 2;
        bool T = 7 != 2;
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("T".into()),
                    value: Expression::Comparison(
                        Box::new(Expression::IntLiteral(7)),
                        CompareOp::GreaterThan,
                        Box::new(Expression::IntLiteral(2))
                    )
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("T".into()),
                    value: Expression::Comparison(
                        Box::new(Expression::IntLiteral(7)),
                        CompareOp::LessThan,
                        Box::new(Expression::IntLiteral(2))
                    )
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("T".into()),
                    value: Expression::Comparison(
                        Box::new(Expression::IntLiteral(7)),
                        CompareOp::GreaterThanOrEqual,
                        Box::new(Expression::IntLiteral(2))
                    )
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("T".into()),
                    value: Expression::Comparison(
                        Box::new(Expression::IntLiteral(7)),
                        CompareOp::LessThanOrEqual,
                        Box::new(Expression::IntLiteral(2))
                    )
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("T".into()),
                    value: Expression::Comparison(
                        Box::new(Expression::IntLiteral(7)),
                        CompareOp::StrictEqual,
                        Box::new(Expression::IntLiteral(2))
                    )
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("T".into()),
                    value: Expression::Comparison(
                        Box::new(Expression::IntLiteral(7)),
                        CompareOp::StrictUnequal,
                        Box::new(Expression::IntLiteral(2))
                    )
                }),
            ]
        }
    );
}

fn parse_if_statement(tokens: &mut Vec<Token>) -> Result<Statement, String> {
    // Ensure 'if' and eat
    match tokens.remove(0).token_type {
        TokenType::If => (),
        _ => return Err("Expected 'if'".to_string()),
    }

    let condition = if tokens[0].token_type == TokenType::LeftParen {
        tokens.remove(0); // Eat '('
        let condition = parse_expression(tokens);
        if tokens[0].token_type != TokenType::RightParen {
            return Err("Expected ')' after if condition".to_string());
        }
        tokens.remove(0); // Eat ')'
        Box::new(condition)
    } else {
        return Err("Expected '(' after 'if'".to_string());
    };

    let mut body = Vec::new();
    if tokens[0].token_type == TokenType::LeftBrace {
        tokens.remove(0); // Eat '{'
        while tokens[0].token_type != TokenType::RightBrace {
            body.push(parse_statement(tokens)?);
        }
        tokens.remove(0); // Eat '}'
    } else {
        return Err("Expected '{' after if condition".to_string());
    }

    let alternative = if tokens
        .get(0)
        .map_or(false, |t| t.token_type == TokenType::Else)
    {
        tokens.remove(0); // Eat 'else'
        let mut alt_body = Vec::new();
        if tokens[0].token_type == TokenType::LeftBrace {
            tokens.remove(0); // Eat '{'
            while tokens[0].token_type != TokenType::RightBrace {
                alt_body.push(parse_statement(tokens)?);
            }
            tokens.remove(0); // Eat '}'
        } else {
            return Err("Expected '{' after 'else'".to_string());
        }
        Some(alt_body)
    } else {
        None
    };

    Ok(Statement::IfStatement(IfStatement {
        condition,
        body,
        alternative,
    }))
}

#[test]
fn assert_parse_if_stmt_empty() {
    let input = r#"
        i32 f() {
            if (7 > 2) {}
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int32,
                identifier: Identifier("f".into()),
                parameters: vec![],
                body: vec![Statement::IfStatement(IfStatement {
                    condition: Box::new(Expression::Comparison(
                        Box::new(Expression::IntLiteral(7)),
                        CompareOp::GreaterThan,
                        Box::new(Expression::IntLiteral(2))
                    )),
                    body: vec![],
                    alternative: None
                })]
            }),]
        }
    );
}

#[test]
fn assert_parse_if_else_stmt_empty() {
    let input = r#"
        i32 f() {
            if (7 > 2) {} else {}
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int32,
                identifier: Identifier("f".into()),
                parameters: vec![],
                body: vec![Statement::IfStatement(IfStatement {
                    condition: Box::new(Expression::Comparison(
                        Box::new(Expression::IntLiteral(7)),
                        CompareOp::GreaterThan,
                        Box::new(Expression::IntLiteral(2))
                    )),
                    body: vec![],
                    alternative: Some(vec![])
                })]
            }),]
        }
    );
}

#[test]
fn assert_parse_if_else_stmt_decls() {
    let input = r#"
        i32 f() {
            if (7 > 2) {
                i32 k = 2;
                return 8;
            } else {
                return 6 - 2;
            }
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int32,
                identifier: Identifier("f".into()),
                parameters: vec![],
                body: vec![Statement::IfStatement(IfStatement {
                    condition: Box::new(Expression::Comparison(
                        Box::new(Expression::IntLiteral(7)),
                        CompareOp::GreaterThan,
                        Box::new(Expression::IntLiteral(2))
                    )),
                    body: vec![
                        Statement::VariableDeclaration(VariableDeclaration {
                            data_type: Type::Int32,
                            identifier: Identifier("k".into()),
                            value: Expression::IntLiteral(2)
                        }),
                        Statement::Return(Expression::IntLiteral(8))
                    ],
                    alternative: Some(vec![Statement::Return(Expression::Binary(
                        Box::new(Expression::IntLiteral(6)),
                        BinaryOp::Sub,
                        Box::new(Expression::IntLiteral(2))
                    ))])
                })]
            }),]
        }
    );
}

#[test]
fn assert_parse_if_else_stmt_branching() {
    let input = r#"
        i32 f() {
            if (7 > 2) {
                if (5 < 9) {} else {}
            }
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int32,
                identifier: Identifier("f".into()),
                parameters: vec![],
                body: vec![Statement::IfStatement(IfStatement {
                    condition: Box::new(Expression::Comparison(
                        Box::new(Expression::IntLiteral(7)),
                        CompareOp::GreaterThan,
                        Box::new(Expression::IntLiteral(2))
                    )),
                    body: vec![Statement::IfStatement(IfStatement {
                        condition: Box::new(Expression::Comparison(
                            Box::new(Expression::IntLiteral(5)),
                            CompareOp::LessThan,
                            Box::new(Expression::IntLiteral(9))
                        )),
                        body: vec![],
                        alternative: Some(vec![])
                    })],
                    alternative: None
                })]
            }),]
        }
    );
}

#[test]
fn assert_parse_if_else_stmt_multi_branching() {
    let input = r#"
        i32 f() {
            if (7 > 2) {
                if (5 < 9) {
                    return 1;
                } else {
                    return 2;
                }
            } else {
                if (5 < 9) {
                    return 3;
                } else {
                    return 4;
                }
            }
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int32,
                identifier: Identifier("f".into()),
                parameters: vec![],
                body: vec![Statement::IfStatement(IfStatement {
                    condition: Box::new(Expression::Comparison(
                        Box::new(Expression::IntLiteral(7)),
                        CompareOp::GreaterThan,
                        Box::new(Expression::IntLiteral(2))
                    )),
                    body: vec![Statement::IfStatement(IfStatement {
                        condition: Box::new(Expression::Comparison(
                            Box::new(Expression::IntLiteral(5)),
                            CompareOp::LessThan,
                            Box::new(Expression::IntLiteral(9))
                        )),
                        body: vec![Statement::Return(Expression::IntLiteral(1))],
                        alternative: Some(vec![Statement::Return(Expression::IntLiteral(2))])
                    })],
                    // outer else
                    alternative: Some(vec![Statement::IfStatement(IfStatement {
                        condition: Box::new(Expression::Comparison(
                            Box::new(Expression::IntLiteral(5)),
                            CompareOp::LessThan,
                            Box::new(Expression::IntLiteral(9))
                        )),
                        body: vec![Statement::Return(Expression::IntLiteral(3))],
                        alternative: Some(vec![Statement::Return(Expression::IntLiteral(4))])
                    })])
                })]
            }),]
        }
    );
}

fn parse_while_statement(tokens: &mut Vec<Token>) -> Result<Statement, String> {
    // Ensure is 'while'
    match tokens.remove(0).token_type {
        TokenType::While => (),
        _ => return Err("Expected 'while'".to_string()),
    }

    let condition = if tokens[0].token_type == TokenType::LeftParen {
        tokens.remove(0); // Eat '('
        let condition = parse_expression(tokens);
        if tokens[0].token_type != TokenType::RightParen {
            return Err("Expected ')' after while condition".to_string());
        }
        tokens.remove(0); // Eat ')'
        Box::new(condition)
    } else {
        return Err("Expected '(' after 'while'".to_string());
    };

    let mut body = Vec::new();
    if tokens[0].token_type == TokenType::LeftBrace {
        tokens.remove(0); // Eat '{'
        while tokens[0].token_type != TokenType::RightBrace {
            let statement = parse_statement(tokens)?;
            body.push(statement);
        }
        tokens.remove(0); // Eat '}'
    } else {
        return Err("Expected '{' after while condition".to_string());
    }

    Ok(Statement::WhileStatement(WhileStatement {
        condition,
        body,
    }))
}

#[test]
fn assert_parse_while_stmt_empty() {
    // We only need to check this once bc it's also statement like if..else
    // but with no alternative, so branching only.
    let input = r#"
        i32 f() {
            while (7 > 2) {}
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int32,
                identifier: Identifier("f".into()),
                parameters: vec![],
                body: vec![Statement::WhileStatement(WhileStatement {
                    condition: Box::new(Expression::Comparison(
                        Box::new(Expression::IntLiteral(7)),
                        CompareOp::GreaterThan,
                        Box::new(Expression::IntLiteral(2))
                    )),
                    body: vec![]
                })]
            }),]
        }
    );
}

fn parse_for_loop(tokens: &mut Vec<Token>) -> Result<Statement, String> {
    match tokens.remove(0).token_type {
        TokenType::For => (), // Eat 'for'
        _ => return Err("Expected 'for'".to_string()),
    }

    if tokens.remove(0).token_type != TokenType::LeftParen {
        return Err("Expected '(' after 'for'".to_string()); // Eat '('
    }

    let init = parse_variable_declaration(tokens);
    if tokens.remove(0).token_type != TokenType::Semicolon {
        return Err("Expected ';' after test condition in for loop".to_string());
    }

    let test = parse_expression(tokens);
    if tokens.remove(0).token_type != TokenType::Semicolon {
        return Err("Expected ';' after test condition in for loop".to_string());
    }

    let update = parse_assignment_expression(tokens)?;

    if tokens.remove(0).token_type != TokenType::RightParen {
        return Err("Expected ')' after for loop clauses".to_string()); // Eat ')'
    }

    let mut body = Vec::new();
    if tokens[0].token_type == TokenType::LeftBrace {
        tokens.remove(0); // Eat '{'
        while tokens[0].token_type != TokenType::RightBrace {
            let statement = parse_statement(tokens)?;
            body.push(statement);
        }
        tokens.remove(0); // Eat '}'
    } else {
        return Err("Expected '{' to start the body of the for loop".to_string());
    }

    Ok(Statement::ForLoop(ForLoop {
        init,
        test,
        update,
        body: Some(body),
    }))
}

#[test]
fn assert_parse_for_stmt_empty() {
    let input = r#"
        i32 f() {
            for (i32 i = 0; i < 10; i += 1) {}
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int32,
                identifier: Identifier("f".into()),
                parameters: vec![],
                body: vec![Statement::ForLoop(ForLoop {
                    init: VariableDeclaration {
                        data_type: Type::Int32,
                        identifier: Identifier("i".into()),
                        value: Expression::IntLiteral(0)
                    },
                    test: Expression::Comparison(
                        Box::new(Expression::Variable(Identifier("i".into()))),
                        CompareOp::LessThan,
                        Box::new(Expression::IntLiteral(10))
                    ),
                    update: Expression::Assignment(Box::new(AssignmentExpression {
                        left: Identifier("i".into()),
                        operator: AssignmentOperator::AddAssign,
                        right: Box::new(Expression::IntLiteral(1))
                    })),
                    body: Some(vec![])
                })]
            }),]
        }
    );
}

// fn parse_statement(tokens: &mut Vec<Token>) -> Result<Statement, String> {
//     match tokens.get(0).map(|t| &t.token_type) {
//         Some(TokenType::If) => parse_if_statement(tokens),
//         Some(TokenType::While) => parse_while_statement(tokens),
//         Some(TokenType::Return) => {
//             tokens.remove(0); // Eat 'return'
//             let expression = parse_expression(tokens);
//             if tokens[0].token_type == TokenType::Semicolon {
//                 tokens.remove(0); // Eat ';'
//             } else {
//                 return Err("Expected ';' after return statement".to_string());
//             }
//             Ok(Statement::Return(expression))
//         }
//         Some(TokenType::Identifier(_)) => {
//             let statement = parse_variable_declaration(tokens);
//             if tokens
//                 .get(0)
//                 .map_or(false, |t| t.token_type == TokenType::Semicolon)
//             {
//                 tokens.remove(0); // Eat ';'
//             }
//             Ok(Statement::VariableDeclaration(statement))
//         }
//         _ => Err("Unrecognized statement".to_string()),
//     }
// }

fn parse_statement(tokens: &mut Vec<Token>) -> Result<Statement, String> {
    match tokens.get(0).map(|t| &t.token_type) {
        Some(TokenType::If) => parse_if_statement(tokens),
        Some(TokenType::While) => parse_while_statement(tokens),
        Some(TokenType::For) => parse_for_loop(tokens),
        Some(TokenType::Return) => {
            tokens.remove(0); // Eat 'return'
            let expression = parse_expression(tokens);
            if tokens[0].token_type == TokenType::Semicolon {
                tokens.remove(0); // Eat ';'
            }
            Ok(Statement::Return(expression))
        }
        Some(TokenType::Identifier(_)) => {
            if let Some(token) = tokens.get(1) {
                match token.token_type {
                    TokenType::Equal
                    | TokenType::PlusEqual
                    | TokenType::MinusEqual
                    | TokenType::MulEqual
                    | TokenType::DivEqual => {
                        // Parse as an assignment expression.
                        let expression = parse_assignment_expression(tokens)?;
                        if tokens
                            .get(0)
                            .map_or(false, |t| t.token_type == TokenType::Semicolon)
                        {
                            tokens.remove(0); // Eat ';'
                        } else {
                            return Err("Expected ';' after assignment".to_string());
                        }
                        Ok(Statement::ExpressionStatement(expression))
                    }

                    // Likely a call expr, routes to parse_call_expression
                    // @todo: fix the '<' issue
                    TokenType::LeftParen | TokenType::Lt => {
                        let expression = parse_call_expression(tokens)?;
                        if tokens
                            .get(0)
                            .map_or(false, |t| t.token_type == TokenType::Semicolon)
                        {
                            tokens.remove(0); // Consume ';'
                        }
                        Ok(Statement::ExpressionStatement(expression))
                    }

                    // var decl statement
                    _ => {
                        let statement = parse_variable_declaration(tokens);
                        if tokens
                            .get(0)
                            .map_or(false, |t| t.token_type == TokenType::Semicolon)
                        {
                            tokens.remove(0); // Eat ';'
                        }
                        Ok(Statement::VariableDeclaration(statement))
                    }
                }
            } else {
                Err("Unexpected end of tokens".to_string())
            }
        }
        _ => Err("Unrecognized statement".to_string()),
    }
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

    // while tokens[0].token_type != TokenType::RightBrace {
    //     match tokens[0].token_type {
    //         TokenType::Identifier(_) => {
    //             let variable_declaration = parse_variable_declaration(tokens);
    //             body.push(Statement::VariableDeclaration(variable_declaration));
    //             if tokens[0].token_type == TokenType::Semicolon {
    //                 tokens.remove(0); // Eat ';'
    //             }
    //         }

    //         TokenType::Return => {
    //             tokens.remove(0); // Eat 'return'
    //             let expression = parse_expression(tokens);
    //             body.push(Statement::Return(expression));
    //             if tokens[0].token_type == TokenType::Semicolon {
    //                 tokens.remove(0); // Eat ';'
    //             }
    //         }

    //         TokenType::EOF | TokenType::Semicolon => break, // /!\
    //         _ => panic!("Unexpected token in function body"),
    //     }
    // }

    while tokens[0].token_type != TokenType::RightBrace {
        let statement = parse_statement(tokens)?;
        body.push(statement);
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
        i32 f() {}
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int32,
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
        i32 f(i32 a, bool b, i32 c, bool d) {
            i32 k = 7 + 10;
        }
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![Declaration::Function(FunctionDeclaration {
                return_type: Type::Int32,
                identifier: Identifier("f".into()),
                parameters: vec![
                    (Type::Int32, Identifier("a".into())),
                    (Type::Bool, Identifier("b".into())),
                    (Type::Int32, Identifier("c".into())),
                    (Type::Bool, Identifier("d".into())),
                ],
                body: vec![Statement::VariableDeclaration(VariableDeclaration {
                    data_type: Type::Int32,
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
        bool gt(i32 a, i32 b) {
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
                    (Type::Int32, Identifier("a".into())),
                    (Type::Int32, Identifier("b".into())),
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
        i32 i = 0;
        i32 i(i32 j) {
            i32 i;
        }
        i32 j = 0;
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    assert_eq!(
        program,
        Program {
            declarations: vec![
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
                    identifier: Identifier("i".into()),
                    value: Expression::IntLiteral(0)
                }),
                Declaration::Function(FunctionDeclaration {
                    return_type: Type::Int32,
                    identifier: Identifier("i".into()),
                    parameters: vec![(Type::Int32, Identifier("j".into())),],
                    body: vec![Statement::VariableDeclaration(VariableDeclaration {
                        data_type: Type::Int32,
                        identifier: Identifier("i".into()),
                        value: Expression::Uninit
                    })]
                }),
                Declaration::Variable(VariableDeclaration {
                    data_type: Type::Int32,
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

            TokenType::Asm => match parse_asm_block(tokens) {
                Ok(asm_block) => declarations.push(Declaration::AsmBlock(asm_block)),
                Err(e) => panic!("Failed to parse asm block: {}", e),
            },

            _ => {
                tokens.remove(0); // /!\ skip unexpected tokens.
            }
        }
    }

    Program { declarations }
}

// The visitor implementation we'll be using will allow multiple mutable references to a `Program`--no clone().
// - A first pass can enforce some known semantics and maybe collect local imports into a map for linking.
// - A separate pass for the type checker, after resolving imports.
// - A second pass will hold the printer for outputting to a `.wat` module.
// - For now, we'll use an asm block for imports.
trait ProgramVisitor {
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

// We use a Vec<String> because we can later iterate and collect into a
// string interning structure. Or maybe it should be an AsRef<T>?
#[derive(Debug, Clone)]
struct Printer {
    indent_level: u8,
    lines: Vec<String>,
}

impl Printer {
    fn new() -> Self {
        Printer { lines: Vec::new(), indent_level: 0 }
    }

    fn def_mod(&mut self) {
        self.lines.push("(module".to_string());
    }

    fn end_mod(&mut self) {
        self.lines.push(")".to_string());
    }

    fn def_global_var(&mut self, name: &str, var_type: &str, value: &str) {
        self.lines.push(format!(
            "  (global ${} ({} {}) ({}))",
            name, var_type, "const", value
        ));
    }

    // This just DEFINES (!), no assignment until Expression::Uninit is checked.
    fn def_local_var(&mut self, name: &str, var_type: &str) {
        self.lines
            .push(format!("{}(local ${} {})", "  ".repeat(self.indent_level.into()), name, var_type));
    }

    fn def_func(&mut self, name: &str, params: Vec<(String, String)>, return_type: Option<String>) {
        let params_str = params
            .into_iter()
            .map(|(type_, id)| format!("(param ${} {})", id, type_))
            .collect::<Vec<_>>()
            .join(" ");
        let return_str =
            return_type.map_or(String::new(), |r_type| format!(" (result {})", r_type));
        self.lines
            .push(format!("{}(func ${}{}{})", "  ".repeat(self.indent_level.into()), name, params_str, return_str));
    }

    fn end_func(&mut self) {
        self.lines.push(format!("{})", "  ".repeat(self.indent_level.into())));
    }

    fn to_string(&self) -> String {
        self.lines.join("\n")
    }
}

struct FirstPass;

impl ProgramVisitor for FirstPass {}

struct SecondPass {
    printer: Printer,
}

impl Default for SecondPass {
    fn default() -> Self {
        Self {
            printer: Printer::new(),
        }
    }
}

impl ProgramVisitor for SecondPass {
    fn visit_program(&mut self, program: &mut Program) {
        self.printer.def_mod();
        
        // self.printer.indent_level += 1; /* temp */

        for declaration in &mut program.declarations {
            self.visit_declaration(declaration);
        }

        self.printer.end_mod();
    }

    // We need to determine whether this is a global or local var.
    fn visit_variable_declaration(&mut self, var_decl: &mut VariableDeclaration) {
        self.printer
            .def_local_var(&var_decl.identifier.0, &var_decl.data_type.to_string());
    }

    fn visit_function_declaration(&mut self, _func_decl: &mut FunctionDeclaration) {
        self.printer.indent_level += 1;
        self.printer.def_func(
            &_func_decl.identifier.0,
            _func_decl
                .parameters
                .iter()
                .map(|(t, id)| (t.to_string(), id.0.clone()))
                .collect::<Vec<(String, String)>>(),
            None,
        );

        self.printer.indent_level += 1;

        for statement in &mut _func_decl.body {
            statement.accept(self);
        }

        self.printer.indent_level -= 1;

        self.printer.end_func();
    }
}

fn main() {
    let input = r#"
        i32 a = 5;

        i32 f() {
            i32 z = 7;
            i32 p = 9;
        }
    "#;
    let mut tokens = collect_tokens(input);
    let mut program = parse_program(&mut tokens);
    let mut first_pass_visitor = FirstPass;
    program.accept(&mut first_pass_visitor);
    let mut second_pass_visitor = SecondPass::default();
    program.accept(&mut second_pass_visitor);
    let output = second_pass_visitor.printer.to_string();
    println!("{}", output);
}
