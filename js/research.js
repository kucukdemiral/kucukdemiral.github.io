(function () {
    'use strict';

    var B = BibParser;

    var themeKeywords = {
        robust: [
            'h-infinity', 'h_infinity', 'h_\\infty', 'h∞', 'mathcal h', 'mathcal{h}',
            'sliding mode', 'backstepping', 'lmi', 'linear matrix inequalit',
            'robust control', 'robust delay', 'robust disturbance',
            'gain-scheduling', 'gain scheduling', 'lpv', 'linear parameter varying',
            'actuator saturat', 'lyapunov', 'stabiliz', 'stability criter',
            'feedforward control', 'delay-dependent', 'time-delay', 'time delay',
            'fractional-order', 'fractional order',
            'fuzzy', 'adaptive control', 'adaptive robust',
            'vibration control', 'vibration suppression', 'active vibration',
            'disturbance rejection', 'disturbance attenuation',
            'fault-tolerant', 'fault tolerant',
            'observer design', 'observer-based',
            'output feedback', 'state feedback',
            'pid', 'self-tuning', 'self tuning'
        ],
        mpc: [
            'model predictive', 'predictive control', 'mpc',
            'receding horizon', 'event-triggered', 'event triggered',
            'tube-based', 'tube based',
            'ship roll', 'ship motion', 'roll motion', 'fin control',
            'offset-free', 'offset free',
            'vertical motion'
        ],
        datadriven: [
            'data-driven', 'data driven', 'data-enabled', 'data enabled',
            'deepc', 'deep reinforcement',
            'learning-based', 'learning based',
            'neural network', 'radial basis',
            'q-learning', 'firefly algorithm', 'optimization algorithm',
            'whale optimization', 'metaheuristic',
            'generative ai', 'large language model', 'llm',
            'deep learning', 'load forecasting',
            'pseudo-data', 'fault identifiab'
        ],
        robotics: [
            'quadrotor', 'uav', 'unmanned aerial',
            'underwater', 'auv', 'autonomous underwater',
            'robot', 'mobile robot', 'wheeled',
            'collaborative robot', 'cobot',
            'manipulator', 'pendulum', 'inverted pendulum',
            'teleoperation', 'bilateral',
            'autonomous vehicle', 'motion sickness',
            'linear motor', 'motor drive', 'positioning',
            'stewart platform', 'space vehicle',
            'trajectory tracking'
        ],
        energy: [
            'wind energy', 'wind turbine', 'dfig',
            'solar energy', 'solar', 'photovoltaic',
            'renewable energy', 'smart grid', 'power system',
            'energy conversion', 'load forecasting',
            'demand response', 'energy transition',
            'microgrid', 'dc microgrid',
            'voltage-source inverter', 'false-data injection',
            'electric vehicle', 'ev charging', 'mobile charging',
            'bayesian network', 'condition monitoring', 'reliability assessment',
            'cyber-resilient', 'cyber security',
            'power engineering'
        ],
        biomedical: [
            'prosthetic', 'prosthesis', 'gait',
            'cancer', 'carcinoma', 'therapy', 'tumor', 'tumour',
            'biomedical', 'biomedical signal',
            'suspension system', 'active suspension', 'vehicle suspension'
        ]
    };

    function classifyEntry(entry) {
        var blob = [
            entry.title, entry.abstract, entry.journal,
            entry.booktitle, entry.keywords, entry.note
        ].join(' ').toLowerCase();

        var themes = [];
        for (var theme in themeKeywords) {
            var kws = themeKeywords[theme];
            for (var i = 0; i < kws.length; i++) {
                if (blob.indexOf(kws[i]) !== -1) {
                    themes.push(theme);
                    break;
                }
            }
        }
        return themes;
    }

    function init() {
        fetch('pub.bib')
            .then(function (r) { return r.text(); })
            .then(function (raw) {
                var entries = B.parse(raw);
                entries.sort(function (a, b) {
                    return (parseInt(b.year, 10) || 0) - (parseInt(a.year, 10) || 0);
                });

                var themeEntries = {};
                for (var theme in themeKeywords) {
                    themeEntries[theme] = [];
                }

                entries.forEach(function (e) {
                    var themes = classifyEntry(e);
                    themes.forEach(function (t) {
                        themeEntries[t].push(e);
                    });
                });

                // Update card counts
                document.querySelectorAll('.theme-card').forEach(function (card) {
                    var theme = card.getAttribute('data-theme');
                    var papers = themeEntries[theme] || [];
                    var countBadge = card.querySelector('.theme-count');
                    if (countBadge) {
                        countBadge.textContent = papers.length + ' paper' + (papers.length !== 1 ? 's' : '');
                    }
                    var cta = card.querySelector('.theme-cta');
                    if (cta) {
                        cta.textContent = 'View ' + papers.length + ' publication' + (papers.length !== 1 ? 's' : '') + ' →';
                    }
                });

                // Bind click handlers
                document.querySelectorAll('.theme-card').forEach(function (card) {
                    function handler() {
                        var theme = card.getAttribute('data-theme');
                        openModal(theme, themeEntries[theme] || [], raw);
                    }
                    card.addEventListener('click', handler);
                    card.addEventListener('keydown', function (ev) {
                        if (ev.key === 'Enter' || ev.key === ' ') {
                            ev.preventDefault();
                            handler();
                        }
                    });
                });
            })
            .catch(function (err) {
                console.error('Failed to load pub.bib:', err);
            });
    }

    // --- Modal ---
    var overlay, modalTitle, modalBody, modalClose;

    function openModal(theme, papers, rawBib) {
        overlay = document.getElementById('modalOverlay');
        modalTitle = document.getElementById('modalTitle');
        modalBody = document.getElementById('modalBody');
        modalClose = document.getElementById('modalClose');

        var themeNames = {
            robust: 'Robust & Optimal Control',
            mpc: 'Model Predictive Control',
            datadriven: 'Data-Driven Control',
            robotics: 'Robotics & Autonomous Systems',
            energy: 'Renewable Energy & Smart Grids',
            biomedical: 'Biomedical Engineering'
        };

        modalTitle.textContent = themeNames[theme] || theme;

        var html = '';
        papers.forEach(function (e) {
            var doi = e.doi || '';
            if (doi && !doi.startsWith('http')) doi = 'https://doi.org/' + doi;

            html += '<div class="modal-paper">';
            html += '<div class="modal-paper-year">' + (e.year || '?') + '</div>';
            html += '<div class="modal-paper-content">';
            html += '<h4>' + B.renderLatex(e.title || 'Untitled') + '</h4>';
            html += '<p class="modal-paper-authors">' + B.escHtml(B.formatAuthors(e.author)) + '</p>';

            var venue = B.formatVenue(e);
            if (venue) html += '<p class="modal-paper-venue">' + B.escHtml(venue) + '</p>';

            if (B.isUnderReview(e)) {
                html += '<span class="modal-under-review">Under Review</span> ';
            }

            if (e.abstract) {
                html += '<details class="modal-abstract-details"><summary>Abstract</summary>';
                html += '<p class="modal-paper-abstract">' + B.renderLatex(e.abstract) + '</p>';
                html += '</details>';
            }

            var actions = '';
            if (doi) {
                var linkLabel = 'View Paper';
                if (doi.indexOf('arxiv.org') !== -1) linkLabel = 'Preprint';
                else if (doi.indexOf('amzn.') !== -1 || doi.indexOf('a.co/') !== -1) linkLabel = 'View Book';
                actions += '<a href="' + B.escHtml(doi) + '" target="_blank" rel="noopener" class="modal-paper-doi">' + linkLabel + ' →</a> ';
            }
            if (e.url && e.url !== doi) {
                actions += '<a href="' + B.escHtml(e.url) + '" target="_blank" rel="noopener" class="modal-paper-doi modal-paper-link">Link →</a> ';
            }
            actions += '<button class="modal-paper-doi modal-bib-btn" data-key="' + B.escHtml(e._key) + '">BibTeX</button>';
            html += '<div class="modal-paper-actions">' + actions + '</div>';

            html += '</div></div>';
        });

        if (papers.length === 0) {
            html = '<p style="color:var(--text-light);text-align:center;padding:2rem 0;">No publications found for this theme.</p>';
        }

        modalBody.innerHTML = html;

        // Bind BibTeX copy buttons
        modalBody.querySelectorAll('.modal-bib-btn').forEach(function (btn) {
            btn.addEventListener('click', function (ev) {
                ev.stopPropagation();
                var key = btn.getAttribute('data-key');
                var bib = B.extractRawBibtex(rawBib, key);
                navigator.clipboard.writeText(bib).then(function () {
                    btn.textContent = 'Copied!';
                    setTimeout(function () { btn.textContent = 'BibTeX'; }, 2000);
                });
            });
        });

        overlay.classList.add('active');
        document.body.style.overflow = 'hidden';
        modalClose.focus();
    }

    function closeModalFn() {
        var ov = document.getElementById('modalOverlay');
        if (ov) {
            ov.classList.remove('active');
            document.body.style.overflow = '';
        }
    }

    document.addEventListener('DOMContentLoaded', function () {
        init();

        var mc = document.getElementById('modalClose');
        var ov = document.getElementById('modalOverlay');
        if (mc) mc.addEventListener('click', closeModalFn);
        if (ov) ov.addEventListener('click', function (ev) {
            if (ev.target === ov) closeModalFn();
        });
        document.addEventListener('keydown', function (ev) {
            if (ev.key === 'Escape') closeModalFn();
        });

        // Swipe-to-close for mobile
        (function () {
            var modal = document.querySelector('.modal');
            var dragHandle = document.getElementById('modalDragHandle');
            if (!dragHandle || !modal) return;
            var startY = 0, currentY = 0, dragging = false;
            dragHandle.addEventListener('touchstart', function (e) {
                startY = e.touches[0].clientY;
                currentY = startY;
                dragging = true;
                modal.style.transition = 'none';
            }, { passive: true });
            dragHandle.addEventListener('touchmove', function (e) {
                if (!dragging) return;
                currentY = e.touches[0].clientY;
                var dy = currentY - startY;
                if (dy > 0) { modal.style.transform = 'translateY(' + dy + 'px)'; e.preventDefault(); }
            }, { passive: false });
            dragHandle.addEventListener('touchend', function () {
                if (!dragging) return;
                dragging = false;
                modal.style.transition = '';
                if (currentY - startY > 100) closeModalFn();
                modal.style.transform = '';
            });
        })();
    });
})();
