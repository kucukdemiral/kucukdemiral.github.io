(function () {
    'use strict';

    var B = BibParser;

    function render(entries, rawBib) {
        var container = document.getElementById('pub-list');
        var filterBtns = document.querySelectorAll('.pub-filter-btn');
        var searchInput = document.getElementById('pub-search');
        var countEl = document.getElementById('pub-count');
        var currentFilter = 'all';

        entries.sort(function (a, b) {
            return (parseInt(b.year, 10) || 0) - (parseInt(a.year, 10) || 0);
        });

        function update() {
            var query = (searchInput.value || '').toLowerCase();
            var filtered = entries.filter(function (e) {
                if (currentFilter !== 'all' && B.getCategory(e) !== currentFilter) return false;
                if (query) {
                    var blob = [
                        B.stripLatex(e.title), e.author, e.journal,
                        e.booktitle, e.year, B.stripLatex(e.abstract)
                    ].join(' ').toLowerCase();
                    return blob.indexOf(query) !== -1;
                }
                return true;
            });

            countEl.textContent = filtered.length + ' publication' + (filtered.length !== 1 ? 's' : '');

            container.innerHTML = '';
            var lastYear = null;
            filtered.forEach(function (e, idx) {
                var yr = parseInt(e.year, 10) || 0;
                if (yr !== lastYear) {
                    var yearDiv = document.createElement('div');
                    yearDiv.className = 'pub-year-header';
                    yearDiv.textContent = yr || 'Unknown';
                    container.appendChild(yearDiv);
                    lastYear = yr;
                }

                var card = document.createElement('div');
                card.className = 'pub-card';

                var cat = B.getCategory(e);
                var badge = cat === 'journal' ? 'Journal' : cat === 'conference' ? 'Conference' : cat === 'book' ? 'Book' : 'Other';
                var badgeClass = 'pub-badge pub-badge-' + cat;

                var doi = e.doi || '';
                if (doi && !doi.startsWith('http')) doi = 'https://doi.org/' + doi;

                var titleRendered = B.renderLatex(e.title || 'Untitled');
                if (doi) titleRendered = '<a href="' + B.escHtml(doi) + '" target="_blank" rel="noopener">' + titleRendered + '</a>';

                var noteHtml = '';
                if (B.isUnderReview(e)) {
                    noteHtml = ' <span class="pub-under-review">Under Review</span>';
                }

                var html = '';
                html += '<div class="pub-card-header">';
                html += '<span class="' + badgeClass + '">' + badge + '</span>';
                html += '<span class="pub-year">' + (e.year || '') + '</span>';
                html += '</div>';
                html += '<h3 class="pub-title">' + titleRendered + noteHtml + '</h3>';
                html += '<p class="pub-authors">' + B.escHtml(B.formatAuthors(e.author)) + '</p>';
                var venue = B.formatVenue(e);
                if (venue) html += '<p class="pub-venue"><em>' + B.escHtml(venue) + '</em></p>';

                html += '<div class="pub-actions">';
                if (e.abstract) {
                    html += '<button class="pub-action-btn pub-btn-abstract" data-idx="' + idx + '">Abstract</button>';
                }
                html += '<button class="pub-action-btn pub-btn-bibtex" data-key="' + B.escHtml(e._key) + '">BibTeX</button>';
                if (doi) {
                    html += '<a class="pub-action-btn pub-btn-doi" href="' + B.escHtml(doi) + '" target="_blank" rel="noopener">DOI</a>';
                }
                if (e.url) {
                    html += '<a class="pub-action-btn pub-btn-link" href="' + B.escHtml(e.url) + '" target="_blank" rel="noopener">Link</a>';
                }
                html += '</div>';

                card.innerHTML = html;
                container.appendChild(card);
            });

            container.querySelectorAll('.pub-btn-abstract').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    var fidx = parseInt(btn.getAttribute('data-idx'));
                    var entry = filtered[fidx];
                    showModal('Abstract',
                        '<p class="modal-pub-title">' + B.renderLatex(entry.title) + '</p>' +
                        '<p>' + B.renderLatex(entry.abstract) + '</p>'
                    );
                });
            });

            container.querySelectorAll('.pub-btn-bibtex').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    var key = btn.getAttribute('data-key');
                    var bib = B.extractRawBibtex(rawBib, key);
                    showModal('BibTeX',
                        '<pre class="modal-bibtex">' + B.escHtml(bib) + '</pre>' +
                        '<button class="pub-copy-btn" id="copy-bib-btn">Copy to Clipboard</button>',
                        function () {
                            var copyBtn = document.getElementById('copy-bib-btn');
                            if (copyBtn) {
                                copyBtn.addEventListener('click', function () {
                                    navigator.clipboard.writeText(bib).then(function () {
                                        copyBtn.textContent = 'Copied!';
                                        setTimeout(function () { copyBtn.textContent = 'Copy to Clipboard'; }, 2000);
                                    });
                                });
                            }
                        }
                    );
                });
            });
        }

        filterBtns.forEach(function (btn) {
            btn.addEventListener('click', function () {
                filterBtns.forEach(function (b) { b.classList.remove('active'); });
                btn.classList.add('active');
                currentFilter = btn.getAttribute('data-filter');
                update();
            });
        });

        var debounceTimer;
        searchInput.addEventListener('input', function () {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(update, 200);
        });

        update();
    }

    // --- Modal ---
    function showModal(title, bodyHtml, afterInsert) {
        var existing = document.getElementById('pub-modal-overlay');
        if (existing) existing.remove();

        var overlay = document.createElement('div');
        overlay.id = 'pub-modal-overlay';
        overlay.className = 'pub-modal-overlay';
        overlay.innerHTML =
            '<div class="pub-modal">' +
            '<div class="pub-modal-header">' +
            '<h3>' + B.escHtml(title) + '</h3>' +
            '<button class="pub-modal-close" aria-label="Close">&times;</button>' +
            '</div>' +
            '<div class="pub-modal-body">' + bodyHtml + '</div>' +
            '</div>';

        document.body.appendChild(overlay);
        requestAnimationFrame(function () { overlay.classList.add('visible'); });

        overlay.querySelector('.pub-modal-close').addEventListener('click', closeModal);
        overlay.addEventListener('click', function (ev) {
            if (ev.target === overlay) closeModal();
        });
        document.addEventListener('keydown', escHandler);

        if (afterInsert) afterInsert();
    }

    function closeModal() {
        var overlay = document.getElementById('pub-modal-overlay');
        if (overlay) {
            overlay.classList.remove('visible');
            setTimeout(function () { overlay.remove(); }, 250);
        }
        document.removeEventListener('keydown', escHandler);
    }

    function escHandler(ev) {
        if (ev.key === 'Escape') closeModal();
    }

    // --- Init ---
    document.addEventListener('DOMContentLoaded', function () {
        fetch('pub.bib')
            .then(function (r) { return r.text(); })
            .then(function (raw) {
                var entries = BibParser.parse(raw);
                render(entries, raw);
            })
            .catch(function (err) {
                document.getElementById('pub-list').innerHTML =
                    '<p style="color:#c00">Failed to load publications. ' + B.escHtml(err.message) + '</p>';
            });
    });
})();
