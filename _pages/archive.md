---
layout: single
title: "Archive"
permalink: /archive/
classes: wide          # 폭 넓게(원하면 제거)
show_date: true
author_profile: false
sidebar: false
---

<!-- 검색 입력(테마 내장 폼) -->
{% include search/search_form.html %}

<!-- 태그 칩 -->
<div class="tag-chips">
  <a class="chip" href="{{ '/tags/' | relative_url }}">Show All</a>
  {% assign all_tags = site.tags | sort %}
  {% for t in all_tags %}
    {% assign tag_name = t[0] %}
    <a class="chip" href="{{ '/tags/#' | append: tag_name | slugify | relative_url }}">
      {{ tag_name }} <span class="count">{{ t[1].size }}</span>
    </a>
  {% endfor %}
</div>

<!-- 연도별 그룹핑 목록 -->
{% assign posts_sorted = site.posts | sort: 'date' | reverse %}
{% assign current_year = '' %}

{% for post in posts_sorted %}
  {% assign y = post.date | date: "%Y" %}
  {% if y != current_year %}
  <h2 class="archive-year" id="y{{ y }}">{{ y }}</h2>
  {% assign current_year = y %}
  {% endif %}

  <div class="archive-item">
    <span class="archive-date">{{ post.date | date: "%b %d" }}</span>
    <a class="archive-link" href="{{ post.url | relative_url }}">{{ post.title }}</a>
  </div>
{% endfor %}
