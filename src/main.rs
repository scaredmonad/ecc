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

fn main() {
    let input = r#"
        int a = 5;
    "#;
    let tokens = collect_tokens(input);
    dbg!(tokens);
}
