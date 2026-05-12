# Live interactive Beatles chart gallery

This page embeds interactive visuals directly.

## 1) Song popularity by writer / writers

This chart shows each songwriter's total Spotify stream count, broken down by individual track, using The Beatles' Spotify dataset sourced from Kaggle. Writers are coloured distinctly; hover over a bar to see song-level stream counts. Note the dominance of Lennon–McCartney joint credits versus solo writing attributions.

<iframe
  src="../../_static/beatles/writer_popularity_by_writer.html"
  width="100%"
  height="650"
  style="border: 1px solid #ddd; border-radius: 8px;"
></iframe>

## 2) Sunburst: full Beatles discography by album

This sunburst chart displays the full Beatles discography hierarchically: the outer ring represents individual tracks, the middle ring represents albums, and the inner ring represents era (roughly). Segment size encodes Spotify stream count. **Click any segment to zoom in**; click the centre to zoom back out.

<iframe
  src="../../_static/beatles/sunburst_discography_by_album.html"
  width="100%"
  height="700"
  style="border: 1px solid #ddd; border-radius: 8px;"
></iframe>

## 3) Sunburst: full Beatles discography by writer

The same sunburst structure as above, but the hierarchy is reorganised by **primary songwriter** rather than album. This makes it easier to compare the relative commercial streaming footprint of Lennon, McCartney, Harrison, and Starr across the catalogue.

<iframe
  src="../../_static/beatles/sunburst_discography_by_writer.html"
  width="100%"
  height="700"
  style="border: 1px solid #ddd; border-radius: 8px;"
></iframe>
