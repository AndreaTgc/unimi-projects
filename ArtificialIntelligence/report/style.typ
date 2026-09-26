// Author: Andrea Colombo
// Basic typst style file for writing generic reports
// currently using this for my artificial intelligence project

// Main function that sets the style for the entire document
#let paper(
  title: "",
  authors: (),
  abstract: [],
  keywords: (),
  body
) = {
  set page(
    paper: "us-letter",
    margin: 1in,
    numbering: "1"
  )

  set text(
    font: "New Computer Modern",
    size: 11pt,
    lang: "en"
  )

  set par(justify: true, leading: 0.65em)

  align(center)[
    #set text(size: 18pt, weight: "bold")
    #title
  ]

  align(center)[
    #set text(size: 12pt)
    #authors.join(", ")
  ]

  v(12pt)

  if abstract != [] [
    #set text(size: 10pt)
    *Abstract* --- #abstract

    #if keywords.len() > 0 [
      #v(6pt)
      *Keywords:* #keywords.join(" - ")
    ]
  ]
  
  set heading(numbering: "1.1")
  outline(
    title: [Table of Contents],
    indent: auto,
    depth: 3
  )
  
  pagebreak()

  body
}

// Function for styling code blocks
#let code-block(
  code,
  lang: none,
  caption: none
) = {
  let code-content = block(
    width: 100%,
    fill: rgb("#f5f5f5"),
    inset: 10pt,
    radius: 3pt,
    stroke: 0.5pt + rgb("#e0e0e0"),
    [
      #set align(left)
      #set par(justify: false)
      #set text(font: "Courier New", size: 9pt)
      #if lang != none [
        #text(
          fill: rgb("#666666"),
          weight: "semibold",
          size: 8pt
        )[#lang]
        #v(4pt)
      ]
      #code
    ]
  )
  
  if caption != none {
    figure(
      code-content,
      caption: caption,
      kind: "code",
      supplement: [Listing]
    )
  } else {
    code-content
  }
}

// Hyperparameters display box
#let hyperparams(..params) = {
  block(
    width: 100%,
    fill: rgb("#f8f8f8"),
    inset: 12pt,
    radius: 3pt,
    stroke: 0.5pt + rgb("#cccccc"),
    [
      #set text(size: 10pt)
      #set align(left)
      #grid(
        columns: (auto, 1fr),
        column-gutter: 2em,
        row-gutter: 0.5em,
        ..params.pos().flatten().map(p => (
          text(weight: "medium")[#p.at(0):],
          p.at(1)
        )).flatten()
      )
    ]
  )
}
