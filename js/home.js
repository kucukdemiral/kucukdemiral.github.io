/* Home page: fills the "Recent work" paper lines and the latest-papers list
   from pub.bib. Without JavaScript, or if pub.bib cannot be
   loaded, the page keeps its static text. */
(function () {
    'use strict';

    var B = window.BibParser;
    if (!B || !window.fetch) return;

    var LATEST_COUNT = 4;

    var typeLabels = {
        journal: 'Journal',
        conference: 'Conference',
        chapter: 'Book chapter',
        book: 'Book'
    };

    function paperUrl(e) {
        var doi = (e.doi || '').trim();
        if (doi) {
            return /^https?:\/\//i.test(doi) ? doi : 'https://doi.org/' + doi;
        }
        var url = (e.url || '').trim();
        return /^https?:\/\//i.test(url) ? url : '';
    }

    function venueName(e) {
        var venue = e.journal || e.booktitle || e.publisher || '';
        return B.cleanAccents(venue).replace(/[{}]/g, '').trim();
    }

    function titleHtml(e, className) {
        var title = B.renderLatex(e.title || 'Untitled');
        var url = paperUrl(e);
        if (url) {
            return '<a href="' + B.escHtml(url) + '" target="_blank" rel="noopener">' + title + '</a>';
        }
        return className ? '<span class="' + className + '">' + title + '</span>' : title;
    }

    function fillRecent(entries) {
        var byKey = {};
        entries.forEach(function (e) { byKey[e._key] = e; });

        document.querySelectorAll('[data-bib]').forEach(function (el) {
            var e = byKey[el.getAttribute('data-bib')];
            if (!e) return;

            var venue = venueName(e);
            var when = B.isAccepted(e) ? 'accepted' : (e.year || '');
            // Conference names often start with the year already
            if (when && venue.indexOf(when) !== -1) when = '';
            var tail = [venue, when].filter(Boolean);

            var html = '<span class="recent-label">Recent work</span>' + titleHtml(e, 'recent-title');
            if (tail.length) {
                html += ' <span class="recent-venue">' + B.escHtml(tail.join(', ')) + '</span>';
            }
            el.innerHTML = html;
            el.hidden = false;
        });
    }

    function fillLatest(entries) {
        var list = document.getElementById('latest');
        if (!list || !entries.length) return;

        var html = '';
        entries.slice(0, LATEST_COUNT).forEach(function (e) {
            var badge = B.isAccepted(e)
                ? '<span class="badge badge-accepted">Accepted</span>'
                : '<span class="badge">' + (typeLabels[B.getCategory(e)] || 'Publication') + '</span>';
            var venue = venueName(e);

            html += '<li class="latest-item">';
            html += '<div class="latest-year">' + B.escHtml(e.year || '') + '</div>';
            html += '<div>';
            html += '<h3 class="latest-title">' + titleHtml(e) + '</h3>';
            html += '<p class="latest-meta">' + B.escHtml(B.formatAuthors(e.author)) + '</p>';
            html += '<p class="latest-meta">' + badge;
            if (venue) html += '<span class="latest-venue">' + B.escHtml(venue) + '</span>';
            html += '</p>';
            html += '</div></li>';
        });
        list.innerHTML = html;
    }

    fetch('pub.bib')
        .then(function (r) {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.text();
        })
        .then(function (raw) {
            var entries = B.parse(raw).filter(function (e) { return !B.isHidden(e); });
            entries.sort(B.compareEntries);
            fillRecent(entries);
            fillLatest(entries);
        })
        .catch(function (err) {
            console.error('Failed to load pub.bib:', err);
        });
})();
