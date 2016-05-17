use std::ops::{Add,Div,Mul,Sub};

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
        self.dx = 1.0;
    }

    pub fn exclude(&mut self) {
        self.dx = 0.0;
    }

    pub fn sqrt(&self) -> DualNumber {
        DualNumber {
            x: self.x.sqrt(),
            dx: 0.5*self.dx/self.x.sqrt(),
        }
    }

    pub fn exp(&self) -> DualNumber {
        DualNumber {
            x: self.x.exp(),
            dx: self.dx*self.x.exp(),
        }
    }

    pub fn ln(&self) -> DualNumber {
        DualNumber {
            x: self.x.ln(),
            dx: self.dx/self.x,
        }
    }

    pub fn log(&self, base: f64) -> DualNumber {
        DualNumber {
            x: self.x.log(base),
            dx: self.dx/self.x,
        }
    }

    pub fn sin(&self) -> DualNumber {
        DualNumber {
            x: self.x.sin(),
            dx: self.dx.cos(),
        }
    }

    pub fn cos(&self) -> DualNumber {
        DualNumber {
            x: self.x.cos(),
            dx: -(self.dx.sin()),
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
