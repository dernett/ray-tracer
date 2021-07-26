use ray_tracer::canvas::Canvas;
use ray_tracer::color::Color;

fn main() {
    let mut canvas = Canvas::new(255, 255);
    for row in 0..canvas.height() {
        for col in 0..canvas.width() {
            canvas[row][col] = Color::new(0.2, 0.3, 0.4);
        }
    }
    print!("{}", canvas.to_ppm());
}
