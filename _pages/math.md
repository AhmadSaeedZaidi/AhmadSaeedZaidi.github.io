---
layout: single
permalink: /math/
title: "Math"
sidebar:
  nav: "math"
---

{% assign posts = site.categories.math | sort: "date" | reverse %}
{% for post in posts %}
  {% include archive-single.html type="post" %}
{% endfor %}
