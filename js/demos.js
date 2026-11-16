/* Looping research clips (home page and research theme windows).
   The clips are silent, load only when they come on screen, and stop
   when they leave it. Each has its own pause button, and nothing plays
   by itself when the visitor has asked for reduced motion. Without
   JavaScript the browser's own video controls are shown instead. */
(function () {
    'use strict';

    var CLIPS = [
        {
            id: 'aerial-navigation',
            label: 'Quadrotor in an unmapped warehouse',
            alt: 'Simulation of a quadrotor flying past storage racks in a warehouse to reach a goal point.'
        },
        {
            id: 'quadruped-locomotion',
            label: 'Quadruped on logs and ice',
            alt: 'Simulation of a four-legged robot walking a course of pylons, logs and ice. A blue robot completes the course while a red robot falls.'
        },
        {
            id: 'humanoid-locomotion',
            label: 'Humanoid on planks and wet ground',
            alt: 'Simulation of a humanoid robot walking a course of pylons, planks and wet ground. A blue robot completes the course while a red robot falls.'
        }
    ];

    var ICON_PAUSE = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>';
    var ICON_PLAY = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.2v13.6a.7.7 0 0 0 1.07.6l10.6-6.8a.7.7 0 0 0 0-1.2L9.07 4.6A.7.7 0 0 0 8 5.2z"/></svg>';

    var reduceMotion = !!(window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches);

    var observer = null;
    if ('IntersectionObserver' in window) {
        observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                var video = entry.target;
                if (!entry.isIntersecting) {
                    video.pause();
                } else if (!video._stopped) {
                    start(video);
                }
            });
        }, { threshold: 0.3 });
    }

    function start(video) {
        var attempt = video.play();
        // If the browser refuses to autoplay, the still image and the play button remain
        if (attempt && attempt.catch) attempt.catch(function () {});
    }

    function syncButton(video) {
        var button = video._button;
        if (!button) return;
        button.innerHTML = video.paused ? ICON_PLAY : ICON_PAUSE;
        button.setAttribute('aria-label', (video.paused ? 'Play' : 'Pause') + ' animation: ' + video._label);
    }

    function setUp(video) {
        if (video._ready) return;
        video._ready = true;
        video._label = video.getAttribute('data-label') || 'research clip';
        // Stopped means the visitor chose not to watch; scrolling does not restart it
        video._stopped = reduceMotion;

        video.muted = true;
        video.removeAttribute('controls');

        var button = document.createElement('button');
        button.type = 'button';
        button.className = 'demo-toggle';
        video._button = button;
        video.parentNode.appendChild(button);

        function toggle() {
            if (video.paused) {
                video._stopped = false;
                start(video);
            } else {
                video._stopped = true;
                video.pause();
            }
        }

        button.addEventListener('click', toggle);
        video.addEventListener('click', toggle);
        video.addEventListener('play', function () { syncButton(video); });
        video.addEventListener('pause', function () { syncButton(video); });
        syncButton(video);

        if (observer) {
            observer.observe(video);
        } else if (!video._stopped) {
            start(video);
        }
    }

    function each(root, fn) {
        var videos = (root || document).querySelectorAll('video.demo-video');
        Array.prototype.forEach.call(videos, fn);
    }

    function init(root) {
        each(root, setUp);
    }

    // Stop the clips inside root and stop watching them, for example when the
    // window holding them is closed or its content is about to be replaced
    function release(root) {
        each(root, function (video) {
            video.pause();
            if (observer) observer.unobserve(video);
        });
    }

    // Markup for one clip, for pages that build their content in JavaScript
    function html(clip) {
        var base = 'media/' + clip.id;
        return '<div class="demo-media">' +
            '<video class="demo-video" muted loop playsinline controls preload="none"' +
            ' width="800" height="600" poster="' + base + '.jpg"' +
            ' data-label="' + clip.label + '" aria-label="' + clip.alt + '">' +
            '<source src="' + base + '.mp4" type="video/mp4">' +
            '</video>' +
            '<span class="demo-tag">Simulation</span>' +
            '</div>';
    }

    window.Demos = {
        clips: CLIPS,
        html: html,
        init: init,
        release: release
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () { init(document); });
    } else {
        init(document);
    }
})();
