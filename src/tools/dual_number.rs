use std::ops::{Add,Div,Mul,Sub,Neg};
use std::fmt;

#[derive(Debug,Copy,Clone)]
pub struct DualNumber {
    pub x: f64,
    pub dx: f64,
}

impl DualNumber {
    pub fn new(x: f64) -> DualNumber {
        DualNumber {
            x: x,
            dx: 0.0,
        }
    }

    pub fn select(&mut self) {
        if self.dx == 0.0 {
            self.dx = 1.0
        }
    }

    pub fn exclude(&mut self) {
        self.dx = 0.0
    }

    pub fn recip(self) -> DualNumber {
        DualNumber {
            x: self.x.recip(),
            dx: - self.dx/(self.x * self.x)
        }
    }

    pub fn sqrt(self) -> DualNumber {
        DualNumber {
            x: self.x.sqrt(),
            dx: 0.5*self.dx/self.x.sqrt(),
        }
    }

    pub fn exp(self) -> DualNumber {
        DualNumber {
            x: self.x.exp(),
            dx: self.dx*self.x.exp(),
        }
    }

    pub fn ln(self) -> DualNumber {
        DualNumber {
            x: self.x.ln(),
            dx: self.dx/self.x,
        }
    }

    pub fn log(self, base: f64) -> DualNumber {
        DualNumber {
            x: self.x.log(base),
            dx: self.dx/self.x,
        }
    }

    pub fn sin(self) -> DualNumber {
        DualNumber {
            x: self.x.sin(),
            dx: self.dx.cos(),
        }
    }

    pub fn cos(self) -> DualNumber {
        DualNumber {
            x: self.x.cos(),
            dx: -(self.dx.sin()),
        }
    }

    pub fn max(self, other: DualNumber) -> DualNumber {
        if self.x > other.x {
            self
        } else {
            other
        }
    }

    pub fn min(self, other: DualNumber) -> DualNumber {
        if self.x < other.x {
            self
        } else {
            other
        }
    }

    pub fn abs(self) -> DualNumber {
        DualNumber {
            x: self.x.abs(),
            dx: 0.0,
        }
    }

    pub fn round(self) -> DualNumber {
        DualNumber {
            x: self.x.round(),
            dx: self.dx,
        }
    }

    pub fn powi(self, n: i32) -> DualNumber {
        if self.x == 0.0 {
            DualNumber {
                x: 0.0,
                dx: 0.0,
            }
        } else if self.x < 0.0 {
            let n_f64 = f64::from(n);
            if (n % 2) == 0 {
                (DualNumber::new(n_f64) * (-self).ln()).exp()
            } else {
                -((DualNumber::new(n_f64) * (-self).ln()).exp())
            }
        } else {
            let n_f64 = f64::from(n);
            (DualNumber::new(n_f64) * self.ln()).exp()
        }
    }

    pub fn powf(self, n: f64) -> DualNumber {
        if self.x == 0.0 {
            DualNumber {
                x: 0.0,
                dx: 0.0,
            }
        } else if self.x < 0.0 {
            if (n % 2.0) == 0.0 {
                (DualNumber::new(n) * (-self).ln()).exp()
            } else {
                -((DualNumber::new(n) * (-self).ln()).exp())
            }
        } else {
            (DualNumber::new(n) * self.ln()).exp()
        }
    }
}

impl Add<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn add(self, other: DualNumber) -> DualNumber {
        DualNumber {
            x: self.x + other.x,
            dx: self.dx + other.dx,
        }
    }
}

impl Div<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn div(self, other: DualNumber) -> DualNumber {
        DualNumber {
            x: self.x / other.x,
            dx: (self.dx * other.x - other.dx * self.x) / (other.x * other.x),
        }
    }
}

impl Mul<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn mul(self, other: DualNumber) -> DualNumber {
        DualNumber {
            x: self.x * other.x,
            dx: self.dx * other.x + other.dx * self.x,
        }
    }
}

impl Sub<DualNumber> for DualNumber {
    type Output = DualNumber;

    fn sub(self, other: DualNumber) -> DualNumber {
        DualNumber {
            x: self.x - other.x,
            dx: self.dx - other.dx,
        }
    }
}

impl Neg for DualNumber {
    type Output = DualNumber;

    fn neg(self) -> DualNumber {
        DualNumber {
            x: -self.x,
            dx: -self.dx,
        }
    }
}

impl fmt::Display for DualNumber {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.dx)
    }
}
