function toggleSidebar() {
    document
        .getElementById("sidebar")
        .classList
        .toggle("show");

    document
        .querySelector(".main-content")
        .classList
        .toggle("shift");
}

document.addEventListener("DOMContentLoaded", () => {
    const theme = localStorage.getItem("theme");
    if (theme === "light") {
        document.body.classList.add("light-mode");
    }
});

function toggleTheme() {
    const body = document.body;
    body.classList.toggle("light-mode");
    if (body.classList.contains("light-mode")) {
        localStorage.setItem("theme", "light");
    } else {
        localStorage.setItem("theme", "dark");
    }
}

// Animate Counters
document.addEventListener("DOMContentLoaded", () => {
    const counters = document.querySelectorAll('.counter');
    const speed = 100; // The lower the slower

    counters.forEach(counter => {
        const animate = () => {
            const target = +counter.getAttribute('data-target');
            const count = +counter.innerText;

            const increment = target / speed;

            if (count < target) {
                counter.innerText = Math.ceil(count + increment);
                setTimeout(animate, 20);
            } else {
                counter.innerText = target;
            }
        }
        animate();
    });
});