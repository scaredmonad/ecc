#![allow(dead_code)]
#![allow(unused_variables)]
use std::str::Chars;

#[derive(Debug, Clone, PartialEq)]
enum TokenType {
    Import,
    Export,
    Extern,
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
fn tokenize_input() {
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
    IntLiteral(i64),
    Variable(Identifier),
}

#[derive(Debug, Clone)]
struct VariableDeclaration {
    data_type: Type,
    identifier: Identifier,
    value: Expression,
}

#[derive(Debug, Clone)]
enum Declaration {
    Noop,
    Variable(VariableDeclaration),
}

#[derive(Debug, Clone)]
struct Program {
    declarations: Vec<Declaration>,
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

    // Eat equal op.
    match tokens.remove(0) {
        Token {
            token_type: TokenType::Equal,
            ..
        } => (),
        _ => panic!("Invalid declaration syntax: '=' expected"),
    };

    // Eat RHS expression (value).
    let value = parse_expression(tokens);

    VariableDeclaration {
        data_type: Type::Int, // @todo: derive types with From<String>.
        identifier: Identifier(variable_identifier),
        value,
    }
}

fn parse_expression(tokens: &mut Vec<Token>) -> Expression {
    if tokens.is_empty() {
        panic!("Invalid expression syntax: Empty expression");
    }

    let token = tokens.pop().unwrap();

    match token.token_type {
        TokenType::IntLiteral(_) => {
            let value = token.lexeme.parse().expect("Invalid integer literal");
            Expression::IntLiteral(value)
        }

        TokenType::Identifier(_) => {
            let identifier = Identifier(token.lexeme);
            Expression::Variable(identifier)
        }

        _ => Expression::Variable(Identifier(String::from("temp"))),
    }
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
        int a = 5;
        int b = 7;
        string c = 8;
    "#;
    let mut tokens = collect_tokens(input);
    let program = parse_program(&mut tokens);
    dbg!(program);
}
