#[derive(Debug, Clone, PartialEq)]
pub(crate) enum TokenType {
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
pub(crate) struct Token {
    pub(crate) token_type: TokenType,
    pub(crate) lexeme: String,
}

pub(crate) struct Lexer<'a> {
    pub(crate) input: std::str::Chars<'a>,
    pub(crate) current_char: Option<char>,
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

pub(crate) fn collect_tokens(input: &str) -> Vec<Token> {
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
