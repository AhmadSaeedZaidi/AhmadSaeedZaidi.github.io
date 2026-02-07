---
layout: single
permalink: /math/
title: "Math"
sidebar:
  nav: "math"
---

{% assign ml_posts = site.categories.math %}
{% if ml_posts %}
  {% assign ml_posts = ml_posts | sort: "date" | reverse %}
  {% for post in ml_posts %}
    {% include archive-single.html type="post" %}
  {% endfor %}
{% endif %}
