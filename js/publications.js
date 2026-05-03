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
            i++;
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
                i++;
            } else if (body[i] === '"') {
                i++;
                var vs2 = i;
                while (i < len && body[i] !== '"') {
                    if (body[i] === '\\') i++;
                    i++;
                }
                value = body.substring(vs2, i);
                i++;
            } else {
                var vs3 = i;
                while (i < len && body[i] !== ',' && body[i] !== '}' && body[i] !== '\n') i++;
                value = body.substring(vs3, i).trim();
            }
            if (name && name !== '_type' && name !== '_key') {
                fields[name] = value.trim();
            }
        }
        return fields;
    }

    // --- LaTeX accent cleaning (for author names etc.) ---
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

    function cleanAccents(s) {
        return s
            .replace(/\{\\"\{([a-zA-Z])\}\}/g, function (_, c) { return latexAccent('"', c); })
            .replace(/\{\\([`'^~"c])\{([a-zA-Z])\}\}/g, function (_, a, c) { return latexAccent(a, c); })
            .replace(/\\([`'^~"c])\{([a-zA-Z])\}/g, function (_, a, c) { return latexAccent(a, c); })
            .replace(/\{\\([`'^~"c])([a-zA-Z])\}/g, function (_, a, c) { return latexAccent(a, c); })
            .replace(/\{\\textquoteright\}/g, "’")
            .replace(/\{\\c\s*([a-zA-Z])\}/g, function (_, c) { return latexAccent('c', c); })
            .replace(/\\&/g, '&')
            .replace(/---/g, "—")
            .replace(/--/g, "–")
            .replace(/~/g, ' ');
    }

    // --- LaTeX math to HTML ---
    var symbolMap = {
        'infty': '∞', 'ell': 'ℓ', 'alpha': 'α', 'beta': 'β',
        'gamma': 'γ', 'delta': 'δ', 'mu': 'μ', 'pi': 'π',
        'sigma': 'σ', 'tau': 'τ', 'omega': 'ω', 'lambda': 'λ',
        'times': '×', 'cdot': '·', 'leq': '≤', 'geq': '≥',
        'neq': '≠', 'approx': '≈', 'pm': '±', 'in': '∈',
        'sum': '∑', 'prod': '∏', 'int': '∫', 'partial': '∂',
        'nabla': '∇', 'forall': '∀', 'exists': '∃', 'sqrt': '√',
    };
    var calligraphic = { H: 'ℋ', L: 'ℒ', F: 'ℱ', B: 'ℬ', C: '𝒞', D: '𝒟', E: 'ℰ', G: '𝒢', I: 'ℐ', J: '𝒥', K: '𝒦', M: 'ℳ', N: '𝒩', O: '𝒪', P: '𝒫', Q: '𝒬', R: 'ℛ', S: '𝒮', T: '𝒯', U: '𝒰', V: '𝒱', W: '𝒲', X: '𝒳', Y: '𝒴', Z: '𝒵' };

    function latexMathToHtml(math) {
        return processLatexTokens(math);
    }

    function processLatexTokens(s) {
        var out = '';
        var i = 0;
        while (i < s.length) {
            if (s[i] === '{') {
                var grp = grabGroup(s, i);
                var content = processLatexTokens(grp.text);
                if (grp.end < s.length && s[grp.end] === '_') {
                    var sub = grabGroup(s, grp.end + 1);
                    out += content + '<sub>' + processLatexTokens(sub.text) + '</sub>';
                    i = sub.end;
                } else if (grp.end < s.length && s[grp.end] === '^') {
                    var sup = grabGroup(s, grp.end + 1);
                    out += content + '<sup>' + processLatexTokens(sup.text) + '</sup>';
                    i = sup.end;
                } else {
                    out += content;
                    i = grp.end;
                }
            } else if (s[i] === '_') {
                i++;
                var sub2 = grabGroup(s, i);
                out += '<sub>' + processLatexTokens(sub2.text) + '</sub>';
                i = sub2.end;
            } else if (s[i] === '^') {
                i++;
                var sup2 = grabGroup(s, i);
                out += '<sup>' + processLatexTokens(sup2.text) + '</sup>';
                i = sup2.end;
            } else if (s[i] === '\\') {
                var cmd = s.substring(i).match(/^\\([a-zA-Z]+)/);
                if (cmd) {
                    var name = cmd[1];
                    i += cmd[0].length;
                    if (name === 'mathcal' || name === 'mathscr') {
                        while (i < s.length && s[i] === ' ') i++;
                        if (i < s.length) {
                            var arg = grabGroup(s, i);
                            var letter = arg.text.trim();
                            out += calligraphic[letter] || letter;
                            i = arg.end;
                        }
                    } else if (symbolMap[name]) {
                        out += symbolMap[name];
                    } else {
                        out += name;
                    }
                } else {
                    out += s[i];
                    i++;
                }
            } else {
                out += s[i];
                i++;
            }
        }
        return out;
    }

    function grabGroup(s, i) {
        if (i >= s.length) return { text: '', end: i };
        if (s[i] === '{') {
            var depth = 1;
            var start = i + 1;
            i++;
            while (i < s.length && depth > 0) {
                if (s[i] === '{') depth++;
                else if (s[i] === '}') depth--;
                i++;
            }
            return { text: s.substring(start, i - 1), end: i };
        }
        if (s[i] === '\\') {
            var cmd = s.substring(i).match(/^\\([a-zA-Z]+)/);
            if (cmd) return { text: s.substring(i, i + cmd[0].length), end: i + cmd[0].length };
        }
        return { text: s[i], end: i + 1 };
    }

    // --- Render LaTeX-aware text as safe HTML ---
    function renderLatex(s) {
        if (!s) return '';
        s = cleanAccents(s);
        var parts = s.split(/(\$[^$]+\$)/g);
        var html = '';
        for (var i = 0; i < parts.length; i++) {
            var p = parts[i];
            if (p.charAt(0) === '$' && p.charAt(p.length - 1) === '$') {
                var inner = p.substring(1, p.length - 1);
                html += '<span class="latex-math">' + latexMathToHtml(inner) + '</span>';
            } else {
                p = p.replace(/\\%/g, '%');
                p = p.replace(/\{/g, '').replace(/\}/g, '');
                html += escHtml(p);
            }
        }
        return html;
    }

    // --- Plain-text version for search ---
    function stripLatex(s) {
        if (!s) return '';
        s = cleanAccents(s);
        s = s.replace(/\$([^$]+)\$/g, function (_, m) {
            return m.replace(/\\[a-zA-Z]+/g, ' ').replace(/[_^{}]/g, '');
        });
        s = s.replace(/\\%/g, '%');
        s = s.replace(/[{}]/g, '');
        return s;
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

    function isUnderReview(entry) {
        var note = (entry.note || '').toLowerCase();
        var journal = (entry.journal || '').toLowerCase();
        return /under\s*review/i.test(note) || /under\s*review/i.test(journal);
    }

    function getYear(entry) {
        return parseInt(entry.year, 10) || 0;
    }

    // --- Format authors ---
    function formatAuthors(raw) {
        if (!raw) return '';
        raw = cleanAccents(raw);
        raw = raw.replace(/\{/g, '').replace(/\}/g, '');
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
        if (/under\s*review/i.test(venue)) venue = '';
        if (venue) {
            venue = cleanAccents(venue).replace(/\{/g, '').replace(/\}/g, '');
            parts.push('<em>' + escHtml(venue) + '</em>');
        }
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
        var i = raw.indexOf('{', start);
        var depth = 1;
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
                    var blob = [
                        stripLatex(e.title), e.author, e.journal,
                        e.booktitle, e.year, stripLatex(e.abstract)
                    ].join(' ').toLowerCase();
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

                var titleRendered = renderLatex(e.title || 'Untitled');
                if (doi) titleRendered = '<a href="' + escHtml(doi) + '" target="_blank" rel="noopener">' + titleRendered + '</a>';

                var noteHtml = '';
                if (isUnderReview(e)) {
                    noteHtml = ' <span class="pub-under-review">Under Review</span>';
                }

                var html = '';
                html += '<div class="pub-card-header">';
                html += '<span class="' + badgeClass + '">' + badge + '</span>';
                html += '<span class="pub-year">' + (e.year || '') + '</span>';
                html += '</div>';
                html += '<h3 class="pub-title">' + titleRendered + noteHtml + '</h3>';
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

            container.querySelectorAll('.pub-btn-abstract').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    var fidx = parseInt(btn.getAttribute('data-idx'));
                    var entry = filtered[fidx];
                    showModal('Abstract',
                        '<p class="modal-pub-title">' + renderLatex(entry.title) + '</p>' +
                        '<p>' + renderLatex(entry.abstract) + '</p>'
                    );
                });
            });

            container.querySelectorAll('.pub-btn-bibtex').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    var key = btn.getAttribute('data-key');
                    var bib = extractRawBibtex(rawBib, key);
                    showModal('BibTeX',
                        '<pre class="modal-bibtex">' + escHtml(bib) + '</pre>' +
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
