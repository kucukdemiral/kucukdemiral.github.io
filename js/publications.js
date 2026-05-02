(function () {
    'use strict';

    // --- BibTeX Parser ---
    function parseBibtex(raw) {
        var entries = [];
        var re = /@(\w+)\s*\{([^,]*),/g;
        var match;
        while ((match = re.exec(raw)) !== null) {
            var type = match[1].toLowerCase();
            var key = match[2].trim();
            var start = re.lastIndex;
            var depth = 1;
            var i = start;
            while (i < raw.length && depth > 0) {
                if (raw[i] === '{') depth++;
                else if (raw[i] === '}') depth--;
                i++;
            }
            var body = raw.substring(start, i - 1);
            var fields = parseFields(body);
            fields._type = type;
            fields._key = key;
            entries.push(fields);
        }
        return entries;
    }

    function parseFields(body) {
        var fields = {};
        var i = 0;
        var len = body.length;
        while (i < len) {
            while (i < len && /[\s,]/.test(body[i])) i++;
            if (i >= len) break;
            var nameStart = i;
            while (i < len && body[i] !== '=' && body[i] !== '}') i++;
            if (i >= len || body[i] === '}') break;
            var name = body.substring(nameStart, i).trim().toLowerCase();
            i++; // skip '='
            while (i < len && /\s/.test(body[i])) i++;
            if (i >= len) break;
            var value;
            if (body[i] === '{') {
                i++;
                var depth = 1;
                var vs = i;
                while (i < len && depth > 0) {
                    if (body[i] === '{') depth++;
                    else if (body[i] === '}') depth--;
                    if (depth > 0) i++;
                }
                value = body.substring(vs, i);
                i++; // skip closing }
            } else if (body[i] === '"') {
                i++;
                var vs2 = i;
                while (i < len && body[i] !== '"') {
                    if (body[i] === '\\') i++;
                    i++;
                }
                value = body.substring(vs2, i);
                i++; // skip closing "
            } else {
                var vs3 = i;
                while (i < len && body[i] !== ',' && body[i] !== '}' && body[i] !== '\n') i++;
                value = body.substring(vs3, i).trim();
            }
            if (name && name !== '_type' && name !== '_key') {
                fields[name] = cleanLatex(value.trim());
            }
        }
        return fields;
    }

    function cleanLatex(s) {
        return s
            .replace(/\{\\"\{([a-zA-Z])\}\}/g, function (_, c) { return latexAccent('"', c); })
            .replace(/\{\\([`'^~"c])\{([a-zA-Z])\}\}/g, function (_, a, c) { return latexAccent(a, c); })
            .replace(/\\([`'^~"c])\{([a-zA-Z])\}/g, function (_, a, c) { return latexAccent(a, c); })
            .replace(/\{\\([`'^~"c])([a-zA-Z])\}/g, function (_, a, c) { return latexAccent(a, c); })
            .replace(/\{\\textquoteright\}/g, '’')
            .replace(/\{\\c\s*([a-zA-Z])\}/g, function (_, c) { return latexAccent('c', c); })
            .replace(/\\&/g, '&')
            .replace(/---/g, '—')
            .replace(/--/g, '–')
            .replace(/~/g, ' ')
            .replace(/\{/g, '')
            .replace(/\}/g, '');
    }

    var accentMap = {
        "'a": "á", "'e": "é", "'i": "í", "'o": "ó", "'u": "ú",
        "'A": "Á", "'E": "É", "'I": "Í", "'O": "Ó", "'U": "Ú",
        "`a": "à", "`e": "è", "`i": "ì", "`o": "ò", "`u": "ù",
        "^a": "â", "^e": "ê", "^i": "î", "^o": "ô", "^u": "û",
        "~a": "ã", "~n": "ñ", "~o": "õ",
        "~A": "Ã", "~N": "Ñ", "~O": "Õ",
        '"a': "ä", '"e': "ë", '"i': "ï", '"o': "ö", '"u': "ü",
        '"A': "Ä", '"O': "Ö", '"U': "Ü",
        "co": "ò", "cc": "ç", "cC": "Ç", "cs": "ş", "cS": "Ş",
    };
    function latexAccent(acc, ch) {
        return accentMap[acc + ch] || ch;
    }

    // --- Categorize ---
    function getCategory(entry) {
        var kw = (entry.keywords || '').trim();
        if (kw === 'J') return 'journal';
        if (kw === 'C') return 'conference';
        var t = entry._type;
        if (t === 'article') return 'journal';
        if (t === 'inproceedings' || t === 'conference') return 'conference';
        if (t === 'book' || t === 'inbook') return 'book';
        return 'other';
    }

    function getYear(entry) {
        return parseInt(entry.year, 10) || 0;
    }

    // --- Format authors ---
    function formatAuthors(raw) {
        if (!raw) return '';
        var parts = raw.split(/\s+and\s+/);
        return parts.map(function (a) {
            a = a.trim();
            if (a.indexOf(',') !== -1) {
                var segs = a.split(',');
                return segs.slice(1).join(' ').trim() + ' ' + segs[0].trim();
            }
            return a;
        }).join(', ');
    }

    // --- Format venue ---
    function formatVenue(entry) {
        var parts = [];
        var venue = entry.journal || entry.booktitle || '';
        if (venue) parts.push('<em>' + escHtml(venue) + '</em>');
        if (entry.volume) {
            var v = entry.volume;
            if (entry.number) v += '(' + entry.number + ')';
            parts.push(v);
        }
        if (entry.pages) parts.push('pp. ' + entry.pages);
        return parts.join(', ');
    }

    function escHtml(s) {
        var d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    // --- Get raw BibTeX for an entry ---
    function extractRawBibtex(raw, key) {
        var escaped = key.replace(/[.*+?^${}()|[\]\\\/]/g, '\\$&');
        var re = new RegExp('@\\w+\\s*\\{\\s*' + escaped + '\\s*,', 'g');
        var match = re.exec(raw);
        if (!match) return '';
        var start = match.index;
        var depth = 0;
        var i = raw.indexOf('{', start);
        depth = 1;
        i++;
        while (i < raw.length && depth > 0) {
            if (raw[i] === '{') depth++;
            else if (raw[i] === '}') depth--;
            i++;
        }
        return raw.substring(start, i).trim();
    }

    // --- Render ---
    function render(entries, rawBib) {
        var container = document.getElementById('pub-list');
        var filterBtns = document.querySelectorAll('.pub-filter-btn');
        var searchInput = document.getElementById('pub-search');
        var countEl = document.getElementById('pub-count');
        var currentFilter = 'all';

        entries.sort(function (a, b) { return getYear(b) - getYear(a); });

        function update() {
            var query = (searchInput.value || '').toLowerCase();
            var filtered = entries.filter(function (e) {
                if (currentFilter !== 'all' && getCategory(e) !== currentFilter) return false;
                if (query) {
                    var blob = [e.title, e.author, e.journal, e.booktitle, e.year, e.abstract].join(' ').toLowerCase();
                    return blob.indexOf(query) !== -1;
                }
                return true;
            });

            countEl.textContent = filtered.length + ' publication' + (filtered.length !== 1 ? 's' : '');

            container.innerHTML = '';
            var lastYear = null;
            filtered.forEach(function (e, idx) {
                var yr = getYear(e);
                if (yr !== lastYear) {
                    var yearDiv = document.createElement('div');
                    yearDiv.className = 'pub-year-header';
                    yearDiv.textContent = yr || 'Unknown';
                    container.appendChild(yearDiv);
                    lastYear = yr;
                }

                var card = document.createElement('div');
                card.className = 'pub-card';

                var cat = getCategory(e);
                var badge = cat === 'journal' ? 'Journal' : cat === 'conference' ? 'Conference' : cat === 'book' ? 'Book' : 'Other';
                var badgeClass = 'pub-badge pub-badge-' + cat;

                var doi = e.doi || '';
                if (doi && !doi.startsWith('http')) doi = 'https://doi.org/' + doi;

                var titleHtml = escHtml(e.title || 'Untitled');
                if (doi) titleHtml = '<a href="' + escHtml(doi) + '" target="_blank" rel="noopener">' + titleHtml + '</a>';

                var note = e.note || '';
                var noteHtml = '';
                if (/under\s*review/i.test(note)) {
                    noteHtml = ' <span class="pub-under-review">Under Review</span>';
                }

                var html = '';
                html += '<div class="pub-card-header">';
                html += '<span class="' + badgeClass + '">' + badge + '</span>';
                html += '<span class="pub-year">' + (e.year || '') + '</span>';
                html += '</div>';
                html += '<h3 class="pub-title">' + titleHtml + noteHtml + '</h3>';
                html += '<p class="pub-authors">' + escHtml(formatAuthors(e.author)) + '</p>';
                var venue = formatVenue(e);
                if (venue) html += '<p class="pub-venue">' + venue + '</p>';

                html += '<div class="pub-actions">';
                if (e.abstract) {
                    html += '<button class="pub-action-btn pub-btn-abstract" data-idx="' + idx + '">Abstract</button>';
                }
                html += '<button class="pub-action-btn pub-btn-bibtex" data-key="' + escHtml(e._key) + '">BibTeX</button>';
                if (doi) {
                    html += '<a class="pub-action-btn pub-btn-doi" href="' + escHtml(doi) + '" target="_blank" rel="noopener">DOI</a>';
                }
                if (e.url) {
                    html += '<a class="pub-action-btn pub-btn-link" href="' + escHtml(e.url) + '" target="_blank" rel="noopener">Link</a>';
                }
                html += '</div>';

                card.innerHTML = html;
                container.appendChild(card);
            });

            // bind abstract buttons
            container.querySelectorAll('.pub-btn-abstract').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    var fidx = parseInt(btn.getAttribute('data-idx'));
                    showModal('Abstract', '<p class="modal-pub-title">' + escHtml(filtered[fidx].title) + '</p><p>' + escHtml(filtered[fidx].abstract) + '</p>');
                });
            });

            // bind bibtex buttons
            container.querySelectorAll('.pub-btn-bibtex').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    var key = btn.getAttribute('data-key');
                    var bib = extractRawBibtex(rawBib, key);
                    showModal('BibTeX', '<pre class="modal-bibtex">' + escHtml(bib) + '</pre><button class="pub-copy-btn" id="copy-bib-btn">Copy to Clipboard</button>', function () {
                        var copyBtn = document.getElementById('copy-bib-btn');
                        if (copyBtn) {
                            copyBtn.addEventListener('click', function () {
                                navigator.clipboard.writeText(bib).then(function () {
                                    copyBtn.textContent = 'Copied!';
                                    setTimeout(function () { copyBtn.textContent = 'Copy to Clipboard'; }, 2000);
                                });
                            });
                        }
                    });
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
            '<h3>' + escHtml(title) + '</h3>' +
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
                var entries = parseBibtex(raw);
                render(entries, raw);
            })
            .catch(function (err) {
                document.getElementById('pub-list').innerHTML =
                    '<p style="color:#c00">Failed to load publications. ' + escHtml(err.message) + '</p>';
            });
    });
})();
