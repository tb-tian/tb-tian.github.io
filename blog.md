---
layout: default
title: Blog
permalink: /blog/
---

# Blog

<ul>
{% for post in site.posts %}
  <li>
    <time>{{ post.date | date: "%b %-d, %Y" }}</time> -
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </li>
{% endfor %}
</ul>
