(function () {
    'use strict';

    var B = BibParser;

    var themeMeta = {
        robust: {
            eyebrow: '01 · Control Theory',
            name: 'Robust & Optimal Control'
        },
        mpc: {
            eyebrow: '02 · Predictive Optimisation',
            name: 'Model Predictive Control'
        },
        datadriven: {
            eyebrow: '03 · Learning & Safety',
            name: 'Data-Driven & Conformal Control'
        },
        robotics: {
            eyebrow: '04 · Autonomous Machines',
            name: 'Robotics & Autonomous Systems'
        },
        energy: {
            eyebrow: '05 · Clean Infrastructure',
            name: 'Renewable Energy & Smart Grids'
        },
        biomedical: {
            eyebrow: '06 · Physiological Control',
            name: 'Biomedical Engineering'
        }
    };

    var themeKeywords = {
        robust: [
            'h-infinity', 'h_infinity', 'h_\\infty', 'h∞', 'mathcal h', 'mathcal{h}',
            'integral quadratic', 'iqc',
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
            'pid control', 'pid controller', 'self-tuning', 'self tuning',
            'ell_', 'h sonsuz', 'position feedback', 'lmis',
            'dayanıklı', 'doyumlu eyleyici', 'matris eşitsizlik',
            'kazanç programlama', 'titreşim'
        ],
        mpc: [
            'model predictive', 'predictive control', 'mpc',
            'receding horizon',
            'offset-free', 'offset free', 'öngörülü'
        ],
        datadriven: [
            'data-driven', 'data driven', 'data-enabled', 'data enabled',
            'deepc', 'koopman', 'conformal', 'edmd', 'deep reinforcement',
            'learning-based', 'learning based',
            'neural network', 'radial basis',
            'q-learning', 'firefly algorithm',
            'whale optimization', 'metaheuristic',
            'generative ai', 'large language model',
            'deep learning', 'load forecasting',
            'pseudo-data', 'fault identifiab', 'trafik tahmin'
        ],
        robotics: [
            'quadrotor', 'uav', 'unmanned aerial',
            'underwater', 'auv', 'autonomous underwater',
            'robot', 'legged', 'humanoid', 'mobile robot', 'wheeled',
            'collaborative robot', 'cobot',
            'manipulator', 'pendulum', 'inverted pendulum',
            'teleoperation', 'bilateral',
            'autonomous vehicle', 'motion sickness',
            'linear motor', 'motor drive', 'position control',
            'stewart platform', 'space vehicle',
            'trajectory tracking'
        ],
        energy: [
            'wind energy', 'wind turbine', 'dfig',
            'solar energy', 'photovoltaic',
            'renewable energy', 'smart grid', 'power system',
            'energy conversion', 'load forecasting',
            'demand response', 'energy transition',
            'microgrid', 'dc microgrid',
            'voltage-source inverter', 'false-data injection',
            'electric vehicle', 'ev charging', 'mobile charging',
            'condition monitoring', 'reliability assessment',
            'power engineering'
        ],
        biomedical: [
            'prosthetic', 'prosthesis', 'gait',
            'cancer', 'carcinoma', 'tumor', 'tumour',
            'biomedical', 'biomedical signal'
        ]
    };

    var wordBoundaryKeywords = ['mpc', 'pid', 'llm', 'solar', 'lmi', 'lmis', 'iqc', 'edmd'];

    function matchesKeyword(blob, kw) {
        if (wordBoundaryKeywords.indexOf(kw) !== -1) {
            var re = new RegExp('(?:^|[^a-z])' + kw + '(?:[^a-z]|$)');
            return re.test(blob);
        }
        return blob.indexOf(kw) !== -1;
    }

    function classifyEntry(entry) {
        var blob = B.cleanAccents([
            entry.title, entry.abstract, entry.journal,
            entry.booktitle, entry.keywords, entry.note
        ].join(' ')).replace(/[{}]/g, '').toLowerCase();

        var themes = [];
        for (var theme in themeKeywords) {
            var kws = themeKeywords[theme];
            for (var i = 0; i < kws.length; i++) {
                if (matchesKeyword(blob, kws[i])) {
                    themes.push(theme);
                    break;
                }
            }
        }
        return themes;
    }

    function updateHeroMetrics(entries) {
        var totalEl = document.getElementById('statTotal');
        var journalsEl = document.getElementById('statJournals');
        var spanEl = document.getElementById('statSpan');

        if (totalEl) totalEl.textContent = String(entries.length);

        var journals = entries.filter(function (e) { return B.getCategory(e) === 'journal'; }).length;
        if (journalsEl) journalsEl.textContent = String(journals);

        var years = entries.map(function (e) { return parseInt(e.year, 10); }).filter(Boolean);
        if (spanEl && years.length) {
            spanEl.innerHTML = Math.min.apply(null, years) + '&ndash;' + Math.max.apply(null, years);
        }
    }

    function renderPage(raw) {
        var entries = raw ? B.parse(raw).filter(function (e) { return !B.isHidden(e); }) : [];
        entries.sort(B.compareEntries);

        if (entries.length) updateHeroMetrics(entries);

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

        // Update card counts and latest paper preview (distinct flagship paper per card)
        var usedLatestKeys = {};
        document.querySelectorAll('.theme-card').forEach(function (card) {
            var theme = card.getAttribute('data-theme');
            var papers = themeEntries[theme] || [];
            var jCount = papers.filter(function (p) { return B.getCategory(p) === 'journal'; }).length;

            var countBadge = card.querySelector('.theme-count');
            if (countBadge) {
                countBadge.textContent = papers.length
                    ? (papers.length + ' publications · ' + jCount + ' journal')
                    : 'View theme';
            }

            var latestEl = card.querySelector('[data-latest-for="' + theme + '"]');
            if (latestEl && papers.length > 0) {
                var newest = papers[0];
                for (var k = 0; k < papers.length; k++) {
                    if (!usedLatestKeys[papers[k]._key]) {
                        newest = papers[k];
                        break;
                    }
                }
                usedLatestKeys[newest._key] = true;
                var yr = newest.year ? ' (' + B.escHtml(newest.year) + ')' : '';
                latestEl.innerHTML = '<span class="theme-latest-label">Latest' + yr + '</span>' +
                    B.renderLatex(newest.title || '');
                latestEl.hidden = false;
            }
        });

        // Bind click and keyboard handlers
        document.querySelectorAll('.theme-card').forEach(function (card) {
            function handler() {
                var theme = card.getAttribute('data-theme');
                openModal(theme, themeEntries[theme] || [], raw || '');
            }
            card.addEventListener('click', handler);
            card.addEventListener('keydown', function (ev) {
                if (ev.key === 'Enter' || ev.key === ' ') {
                    ev.preventDefault();
                    handler();
                }
            });
        });

        // Open a theme straight from a link such as research.html#energy
        var hashTheme = (window.location.hash || '').replace('#', '');
        if (Object.prototype.hasOwnProperty.call(themeEntries, hashTheme)) {
            openModal(hashTheme, themeEntries[hashTheme], raw || '');
        }
    }

    function init() {
        var loader = B.loadBib ? B.loadBib() : fetch('pub.bib').then(function (r) { return r.text(); });
        loader
            .then(function (raw) {
                renderPage(raw);
            })
            .catch(function (err) {
                console.error('Failed to load pub.bib:', err);
                renderPage('');
            });
    }

    // --- Clips of current work (js/demos.js) ---
    var demoThemes = { robust: true, mpc: true, datadriven: true, robotics: true };

    function demosHtml() {
        var D = window.Demos;
        if (!D) return '';

        var html = '<section class="modal-demos" aria-label="Current work in motion">';
        html += '<h4>Current work in motion</h4>';
        html += '<p class="modal-demos-note">Switched Koopman&ndash;conformal control under random network delays and nonlinear data-driven predictive control, shown in simulation. Manuscripts in preparation.</p>';
        html += '<div class="modal-demos-row">';
        D.clips.forEach(function (clip) {
            html += '<figure class="modal-demo">' + D.html(clip) +
                '<figcaption>' + clip.label + '</figcaption></figure>';
        });
        html += '</div>';
        html += '<p class="modal-demos-key">';
        html += '<span class="key"><span class="key-dot key-blue"></span>Proposed data-driven controller</span>';
        html += '<span class="key"><span class="key-dot key-ring"></span>Adaptive 3D conformal safety cage</span>';
        html += '<span class="key"><span class="key-dot key-red"></span>Conventional / nominal MPC baseline</span>';
        html += '</p>';
        html += '</section>';
        return html;
    }

    var typeBadgeLabels = {
        journal: { label: 'Journal', cls: '' },
        conference: { label: 'Conference', cls: ' modal-type-conference' },
        chapter: { label: 'Book Chapter', cls: ' modal-type-book' },
        book: { label: 'Book', cls: ' modal-type-book' }
    };

    function renderPaperCards(papers) {
        if (!papers.length) {
            return '<p class="modal-empty">No matching publications found in this theme.</p>';
        }

        var html = '';
        papers.forEach(function (e) {
            var doi = e.doi || '';
            if (doi && !doi.startsWith('http')) doi = 'https://doi.org/' + doi;

            var cat = B.getCategory(e);
            var tInfo = typeBadgeLabels[cat] || { label: 'Publication', cls: '' };

            html += '<div class="modal-paper">';
            html += '<div class="modal-paper-year">' + B.escHtml(e.year || '?') + '</div>';
            html += '<div class="modal-paper-content">';

            html += '<div class="modal-paper-badges">';
            html += '<span class="modal-type-badge' + tInfo.cls + '">' + tInfo.label + '</span>';
            if (B.isUnderReview(e)) {
                html += '<span class="modal-under-review">Under Review</span>';
            } else if (B.isAccepted(e)) {
                html += '<span class="modal-accepted">Accepted</span>';
            }
            html += '</div>';

            html += '<h4>' + B.renderLatex(e.title || 'Untitled') + '</h4>';
            html += '<p class="modal-paper-authors">' + B.escHtml(B.formatAuthors(e.author)) + '</p>';

            var venue = B.formatVenue(e);
            if (venue) html += '<p class="modal-paper-venue">' + B.escHtml(venue) + '</p>';

            if (e.abstract) {
                html += '<details class="modal-abstract-details"><summary>Read abstract</summary>';
                html += '<p class="modal-paper-abstract">' + B.renderLatex(e.abstract) + '</p>';
                html += '</details>';
            }

            var actions = '';
            if (doi) {
                var linkLabel = 'View Paper';
                if (doi.indexOf('arxiv.org') !== -1) linkLabel = 'Preprint';
                else if (doi.indexOf('amzn.') !== -1 || doi.indexOf('a.co/') !== -1) linkLabel = 'View Book';
                actions += '<a href="' + B.escHtml(doi) + '" target="_blank" rel="noopener" class="modal-paper-doi">' + linkLabel + ' &rarr;</a> ';
            }
            if (e.url && e.url !== doi) {
                actions += '<a href="' + B.escHtml(e.url) + '" target="_blank" rel="noopener" class="modal-paper-doi modal-paper-link">Link &rarr;</a> ';
            }
            actions += '<button type="button" class="modal-paper-doi modal-bib-btn" data-key="' + B.escHtml(e._key) + '">BibTeX</button>';
            html += '<div class="modal-paper-actions">' + actions + '</div>';

            html += '</div></div>';
        });
        return html;
    }

    // --- Modal ---
    function openModal(theme, papers, rawBib) {
        var overlay = document.getElementById('modalOverlay');
        var modalEyebrow = document.getElementById('modalEyebrow');
        var modalTitle = document.getElementById('modalTitle');
        var modalSubtitle = document.getElementById('modalSubtitle');
        var modalBody = document.getElementById('modalBody');
        var modalClose = document.getElementById('modalClose');

        if (window.Demos) window.Demos.release(modalBody);

        var meta = themeMeta[theme] || { eyebrow: 'Research Theme', name: theme };
        if (modalEyebrow) modalEyebrow.textContent = meta.eyebrow;
        modalTitle.textContent = meta.name;

        var jCount = papers.filter(function (p) { return B.getCategory(p) === 'journal'; }).length;
        var cCount = papers.filter(function (p) { return B.getCategory(p) === 'conference'; }).length;
        var years = papers.map(function (p) { return parseInt(p.year, 10); }).filter(Boolean);
        var spanText = years.length ? (' · ' + Math.min.apply(null, years) + '–' + Math.max.apply(null, years)) : '';
        if (modalSubtitle) {
            modalSubtitle.textContent = papers.length + ' publications (' + jCount + ' journal, ' + cCount + ' conference)' + spanText;
        }

        var html = '';
        if (demoThemes[theme]) html += demosHtml();

        html += '<div class="modal-toolbar">';
        html += '<div class="modal-filters" role="tablist" aria-label="Filter publications by type">';
        html += '<button type="button" class="modal-filter-btn active" data-type="all">All (' + papers.length + ')</button>';
        if (jCount > 0) html += '<button type="button" class="modal-filter-btn" data-type="journal">Journal (' + jCount + ')</button>';
        if (cCount > 0) html += '<button type="button" class="modal-filter-btn" data-type="conference">Conference (' + cCount + ')</button>';
        html += '</div>';
        html += '<input type="search" class="modal-search" id="modalSearch" placeholder="Filter by keyword or year..." aria-label="Filter publications in this theme">';
        html += '</div>';

        html += '<div id="modalPaperList">' + renderPaperCards(papers) + '</div>';

        modalBody.innerHTML = html;
        modalBody.scrollTop = 0;
        if (window.Demos) window.Demos.init(modalBody);

        var activeType = 'all';
        var searchQuery = '';
        var listEl = document.getElementById('modalPaperList');
        var searchEl = document.getElementById('modalSearch');

        function refreshList() {
            var filtered = papers.filter(function (e) {
                if (activeType !== 'all' && B.getCategory(e) !== activeType) return false;
                if (!searchQuery) return true;
                var hay = B.stripLatex([e.title, e.author, e.journal, e.booktitle, e.year].join(' ')).toLowerCase();
                return hay.indexOf(searchQuery) !== -1;
            });
            listEl.innerHTML = renderPaperCards(filtered);
            bindBibButtons(listEl, rawBib);
        }

        modalBody.querySelectorAll('.modal-filter-btn').forEach(function (btn) {
            btn.addEventListener('click', function () {
                modalBody.querySelectorAll('.modal-filter-btn').forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                activeType = btn.getAttribute('data-type');
                refreshList();
            });
        });

        if (searchEl) {
            searchEl.addEventListener('input', function () {
                searchQuery = searchEl.value.trim().toLowerCase();
                refreshList();
            });
        }

        bindBibButtons(listEl, rawBib);

        overlay.classList.add('active');
        document.body.style.overflow = 'hidden';
        modalClose.focus();
    }

    function bindBibButtons(root, rawBib) {
        root.querySelectorAll('.modal-bib-btn').forEach(function (btn) {
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
    }

    function closeModalFn() {
        var ov = document.getElementById('modalOverlay');
        if (ov) {
            ov.classList.remove('active');
            document.body.style.overflow = '';
            if (window.Demos) window.Demos.release(document.getElementById('modalBody'));
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
