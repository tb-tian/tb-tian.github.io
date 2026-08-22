---
layout: default
title: Publications
permalink: /publications/
---

# Publications

<h1> Just a sample publication. Currently trying so hard to get one for myself. </h1>

{% assign years = site.data.publications | map: "year" | uniq | sort | reverse %}

{% for year in years %}
  <h2>{{ year }}</h2>
  <ul class="pubs">
    {% assign pubs = site.data.publications | where: "year", year %}
    {% for pub in pubs %}
      <li>
        <span class="pub-title">{{ pub.title }}</span><br>
        {{ pub.authors }}<br>
        <em>{{ pub.venue }}</em>
        {% if pub.pdf %} · <a href="{{ pub.pdf }}">PDF</a>{% endif %}
        {% if pub.code %} · <a href="{{ pub.code }}">Code</a>{% endif %}
      </li>
    {% endfor %}
  </ul>
{% endfor %}