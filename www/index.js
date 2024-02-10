import * as sim from "lib-simulation-wasm"

let simulation = new sim.Simulation();
document.getElementById('train').onclick = function () {
    console.log(simulation.train());
};
// fix browser pixel scaling
const viewport = document.getElementById('viewport');
const viewportWidth = viewport.width;
const viewportHeight = viewport.height;
const viewportScale = window.devicePixelRatio || 1;
viewport.width = viewportWidth * viewportScale;
viewport.height = viewportHeight * viewportScale;
viewport.style.width = viewportWidth + 'px';
viewport.style.height = viewportHeight + 'px';

const ctxt = viewport.getContext('2d');

// auto scales everything up
ctxt.scale(viewportScale, viewportScale);

ctxt.fillStyle = 'rgb(0, 0, 0)';

CanvasRenderingContext2D.prototype.drawCreature = function (x, y, size, rotation) {
    this.beginPath();

    this.moveTo(
        x - Math.sin(rotation) * size * 1.5,
        y + Math.cos(rotation) * size * 1.5,
    );

    this.lineTo(
        x - Math.sin(rotation + 2.0 / 3.0 * Math.PI) * size,
        y + Math.cos(rotation + 2.0 / 3.0 * Math.PI) * size,
    );

    this.lineTo(
        x - Math.sin(rotation + 4.0 / 3.0 * Math.PI) * size,
        y + Math.cos(rotation + 4.0 / 3.0 * Math.PI) * size,
    );

    this.lineTo(
        x - Math.sin(rotation) * size * 1.5,
        y + Math.cos(rotation) * size * 1.5,
    );

    this.strokeStyle = `rgb(90, 20, 0)`
    this.stroke();

    this.fillStyle = `rgba(0, 255, 255, .5)`
    this.fill();
};

CanvasRenderingContext2D.prototype.drawCircle = function (x, y, radius) {
    this.beginPath();

    // define circle properties  (center, radius, start_range, end_range);
    this.arc(x, y, radius, 0.0, 2.0 * Math.PI);

    this.fillStyle = `rgb(0, 215, 60)`

    this.fill();
};

function redraw() {
    ctxt.clearRect(0, 0, viewportWidth, viewportHeight);

    simulation.step();

    const world = simulation.world();

    for (const resource of world.resources) {
        ctxt.drawCircle(
            resource.x * viewportWidth,
            resource.y * viewportHeight,
            (0.01 / 2.0) * viewportWidth,
        );
    }

    for (const creature of world.creatures) {
        ctxt.drawCreature(
            creature.x * viewportWidth,
            creature.y * viewportHeight,
            0.01 * viewportWidth,
            creature.rotation,
        );
    }

    // requestAnimationFrame() schedules code only for the next frame.
    //
    // Because we want for our simulation to continue forever, we've
    // gotta keep re-scheduling our function:
    requestAnimationFrame(redraw);
}

redraw();