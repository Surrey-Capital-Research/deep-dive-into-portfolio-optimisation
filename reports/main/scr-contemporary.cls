%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% scr-contemporary.cls, v1, 10/12/2025
% Developed by Overleaf, Edited by Surrey Capital Research
% 
% This class file enables authors to prepare 
% manuscripts for submission to Surrey Capital Research using
% the 'contemporary' layout.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
\ProvidesClass{scr-contemporary}[10/12/2025, v1]
\newif\if@numrefs
\newif\if@alpharefs

%% Options for bibliography style
\DeclareOption{num-refs}{
  \@numrefstrue
  \@alpharefsfalse
}
\DeclareOption{alpha-refs}{
  \@numrefsfalse
  \@alpharefstrue
}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ExecuteOptions{num-refs}
\ProcessOptions\relax
\LoadClass[twoside,twocolumn]{article}

\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{ifpdf}
\RequirePackage{microtype}
\RequirePackage{euler}
\RequirePackage{merriweather}
\renewcommand{\ttdefault}{lmtt}
\RequirePackage[italic]{mathastext}
\RequirePackage{relsize}
\RequirePackage{ragged2e}
\RequirePackage{xcolor}
\RequirePackage{colortbl}
\RequirePackage{graphicx}
\RequirePackage{etoolbox,xpatch}
\RequirePackage{environ}
\RequirePackage{lastpage}
\RequirePackage{rotating}
\RequirePackage{lscape}
\RequirePackage{afterpage}
\RequirePackage[absolute]{textpos}
\RequirePackage{url}
\RequirePackage{newfloat}
\RequirePackage{mdframed}
\RequirePackage{titlesec}

\RequirePackage{scrextend}
\changefontsizes[10.5pt]{7pt}

\RequirePackage{calc}
\RequirePackage{geometry}
\geometry{includefoot,
  hmargin=4pc+6.64pt,
  bottom=2pc+10pt,
  top=5pc+6pt,
  headsep=2pc,
  headheight=17.5pt,
  footnotesep=1pc+6pt,
  columnsep=18pt}
\setlength{\parindent}{12pt}

\AddToHook{begindocument/before}{
\RequirePackage[colorlinks,allcolors=black,urlcolor=blue]{hyperref}}

\RequirePackage[noblocks]{authblk}

% Remove superscripts from authors and affiliations
\renewcommand\AB@affilsepx{ and }
\renewcommand\AB@authnote[1]{}
\renewcommand\AB@affilnote[1]{}

% SCR Specific Defaults
\def\@jname{Surrey Capital Research}
\def\@jlogo{scr-logo.pdf}
\def\@secondlogo{} % Optional second logo

\newcommand{\jname}[1]{\def\@jname{#1}}
\newcommand{\jlogo}[1]{\def\@jlogo{#1}}
\newcommand{\secondlogo}[1]{\def\@secondlogo{#1}}

\def\@jyear{20xx}
\def\@jvolume{0}
\def\@jnumber{0}
\def\@jstartpage{1}

\newcommand{\jyear}[1]{\def\@jyear{#1}}
\newcommand{\jvolume}[1]{\def\@jvolume{#1}}
\newcommand{\jnumber}[1]{\def\@jnumber{#1}}
\newcommand{\jstartpage}[1]{\def\@jstartpage{#1}}

\def\@papercat{}
\def\@runningauthor{}
\def\@runningtitle{}
\def\@corraddress{}
\def\@authnotes{}
\definecolor{jcolour}{RGB}{0,70,128} % Default SCR blue color (change RGB values for different color)

\newcommand{\runningauthor}[1]{\def\@runningauthor{#1}}
\newcommand{\runningtitle}[1]{\def\@runningtitle{#1}}
\newcommand{\corraddress}[1]{\def\@corraddress{#1}}
\newcommand{\authnote}[1]{\appto{\@authnotes}{#1\par}}
\newcommand{\papercat}[1]{\def\@papercat{#1}}

% Removed the journal command as it was GigaScience/general specific.

\RequirePackage[flushmargin,bottom,norule]{footmisc}
\appto{\footnotelayout}{\fontsize{6.5pt}{9.5pt}\selectfont}
\deffootnote{1.5em}{1.5em}{\thefootnotemark\enspace}

\RequirePackage{fancyhdr}
\fancyhf{}
\renewcommand{\headrule}{\color{black!15}\rule{\textwidth}{2pt}}
\fancyhead[LE]{%
  \hspace*{12pt}%
  \fontsize{7.8pt}{10pt}\selectfont%
  \color{jcolour}%
  \textbf{\thepage}%
  \hspace{1em}\raisebox{-0.5pt}{|}\hspace{1em}%
  \textit{\@jname}, \@jyear, Vol.~\@jvolume, No.~\@jnumber
}
\fancyhead[RO]{%
  \fontsize{7.8pt}{10pt}\selectfont%
  \color{jcolour}%
  \@runningauthor%
  \hspace{1em}\raisebox{-0.5pt}{|}\hspace{1em}%
  \textbf{\thepage}%
  \hspace*{12pt}%
}

\fancypagestyle{plain}{%
  \fancyhf{}
  \renewcommand{\headrule}{}
  \fancyfoot[R]{%
    \fontsize{7.8pt}{10pt}\selectfont%
    \color{jcolour}%
    \textbf{\thepage}\hspace*{12pt}%
  }
}


\pagestyle{fancy}


\AddToHook{begindocument/end}{
  \newsavebox{\@gscitedetails}
  \newlength{\@gscitedetailsheight}
  \newsavebox{\@gsheadbox}
  \newlength{\@gsheadboxheight}
  \savebox{\@gscitedetails}{%
    \parbox{\textwidth-21pc+4pc+6.64pt}{%
%     \normalfont{\itshape\bfseries\@jname}, \@jyear, \@jstartpage--\pageref{LastPage}\par
%     \vspace{\dimexpr 8pt-1em\relax}
    {\color{black!15}\rule{\linewidth}{2pt}}\par%
    \vspace{\dimexpr 8pt-1em\relax}%
    \fontsize{6.5pt}{9.5pt}\bfseries%
    \textcolor{jcolour}{doi: xx.xxxx/xxxx}\par
    Manuscript in Preparation\par
    \@papercat\par
    \vspace{\dimexpr 8pt-1em\relax}%
    {\color{black!15}\rule{\linewidth}{2pt}}\par
  }}%
  \settototalheight{\@gscitedetailsheight}{\usebox{\@gscitedetails}}%
  %%%%
  \savebox{\@gsheadbox}{%
    \parbox{\textwidth-21pc+4pc+6.64pt}{%
    \normalfont%
    \fontsize{8pt}{9.5pt}\selectfont%
    {\itshape\bfseries\@jname}, \@jyear, \@jstartpage--\pageref{LastPage}\par
    \vspace{\dimexpr 8pt-1.5pt\relax}%
    \usebox{\@gscitedetails}
    }%
  }%
  \settototalheight{\@gsheadboxheight}{\usebox{\@gsheadbox}}%
}

\RequirePackage{alphalph}
\newalphalph{\fnsymbolmult}[mult]{\@fnsymbol}{5}

\newcounter{authorfn}
\setcounter{authorfn}{1}
\newcommand{\authfn}[1]{%
  \fnsymbolmult{#1}%
}

\renewcommand\Authands{ and }
% Author and affiliation formatting with proper spacing
\renewcommand{\Authfont}{\fontsize{13pt}{18pt}\selectfont}
\renewcommand{\Affilfont}{\fontsize{9pt}{13pt}\selectfont}
\setlength{\affilsep}{21pt-9pt}

\AddToHook{begindocument/end}{
\renewcommand{\@maketitle}{%
  \textblockorigin{4pc+6.64pt}{4pc+6pt}
  %%%%
  \begin{textblock*}{21pc}(0pt,0pt)
  % SCR logo
  \ifdefempty{\@jlogo}{}{%
  \IfFileExists{\@jlogo}{%
    \includegraphics[height=1.3\@gscitedetailsheight]{\@jlogo}
  }{%
    \fbox{\begin{minipage}[b][\@gscitedetailsheight]{\@gscitedetailsheight}%
    Placeholder for logo \@jlogo
    \end{minipage}}%
  }}%
  % Second logo (if specified)
  \ifdefempty{\@secondlogo}{}{%
    \hspace{12pt}%
    \IfFileExists{\@secondlogo}{%
      \includegraphics[height=1.3\@gscitedetailsheight]{\@secondlogo}
    }{}%
  }%
  \end{textblock*}
  %%%
  \begin{textblock*}{\textwidth-21pc}(21pc,0pt)
  \usebox{\@gsheadbox}
  \end{textblock*}%
  %%%
  \null\vspace*{\dimexpr 13pc+11pt-5pc-6pt\relax}%
  \raggedright%
  {\fontsize{14pt}{16pt}\selectfont%
    \ifpdf
      \textls[200]{\relsize{-1}\MakeUppercase{\@papercat}}
    \else %%% don't use letterspace in latex+dvi
      \relsize{-1}\MakeUppercase{\@papercat}%
    \fi%
    \par}
  \vspace{\dimexpr 16pt-18pt\relax}
  {\fontsize{18pt}{27pt}\bfseries\color{jcolour}\@title\par}
  \vspace{\dimexpr 16pt-13pt\relax}
  \@author\par
  \vspace{\dimexpr 16pt-3pt-1em\relax}
}
}

\RequirePackage[style]{abstract}
\setlength{\absleftindent}{0pt}
\setlength{\absrightindent}{0pt}
\setlength{\absparindent}{12pt}
\setlength{\absparsep}{0pt}
\setlength{\abstitleskip}{16pt-8pt-\baselineskip}
\renewcommand{\abstractnamefont}{\fontsize{10pt}{12pt}\bfseries\color{jcolour}}
\renewcommand{\abstitlestyle}[1]{{\abstractnamefont #1}}
\renewcommand{\abstracttextfont}{\fontsize{8pt}{11pt}\selectfont}
\preto{\abstract}{%
  {\noindent\color{jcolour!40}\rule{\hsize+4pc+6.64pt}{3pt}}\par
  \vspace*{\dimexpr 23pt-10pt\relax}
}
\appto\abstract{\noindent\raggedright\ignorespaces}
\preto\endabstract{\par}

\newenvironment{keywords}
{\vspace*{\dimexpr 19pt-8pt-\baselineskip\relax}%
\noindent\fontsize{8pt}{11pt}\selectfont\textbf{Key words}: \ignorespaces}%
{\par}

\DeclareFloatingEnvironment[placement=t!,name=kpt]{keypoints}

\xapptocmd{\keypoints}{%
  \begin{mdframed}[linewidth=1pt,linecolor=black,
  innerleftmargin=8pt,innerrightmargin=8pt,
  innertopmargin=16pt-8.2pt,innerbottommargin=6pt]
  {\fontsize{8.2pt}{10pt}\bfseries Key Points\par}
}{}{}
\xpretocmd{\endkeypoints}{\end{mdframed}}{}{}

\xapptocmd{\keypoints*}{%
  \begin{mdframed}[linewidth=1pt,linecolor=black,
  innerleftmargin=8pt,innerrightmargin=8pt,
  innertopmargin=16pt-8.2pt,innerbottommargin=6pt]
  {\fontsize{8.2pt}{10pt}\bfseries Key Points\par}
}{}{}
\xpretocmd{\endkeypoints*}{\end{mdframed}}{}{}


\if@twocolumn
  \NewEnviron{frontmatter}
  {\twocolumn[%
      \BODY%
      \vspace*{\dimexpr 16pt-3pt-1em\relax}
      {\noindent\color{jcolour!40}\rule{\hsize+4pc+6.64pt}{3pt}}\par
      \vspace*{\dimexpr 23pt-1em\relax}]%
    \begin{figure*}[b!]
      \fontsize{6.5pt}{9.5pt}\selectfont%
      \hrule%
      \vspace{\dimexpr 14pt-6.5pt\relax}%
      \textbf{Compiled on:} \today.\par
      Draft manuscript prepared by the author.\par
    \end{figure*}
  }
\else
  \NewEnviron{frontmatter}{%
      \BODY%
      \vspace*{\dimexpr 16pt-3pt-1em\relax}
      {\noindent\color{jcolour!40}\rule{\hsize+4pc+6.64pt}{3pt}}\par
      \vspace*{\dimexpr 23pt-1em\relax}%
    \begin{figure*}[b!]
      \fontsize{6.5pt}{9.5pt}\selectfont%
      \hrule%
      \vspace{\dimexpr 14pt-6.5pt\relax}%
      \textbf{Compiled on:} \today.\par
      Draft manuscript prepared by the author.\par
    \end{figure*}
  }
\fi

\setcounter{secnumdepth}{3}
\titleformat*{\section}{\fontsize{10pt}{12pt}\bfseries\color{jcolour}}
\titlespacing*{\section}{0pt}{26.5pt-1em}{16pt-1em}
\titleformat*{\subsection}{\fontsize{8.5pt}{11pt}\bfseries\color{jcolour}}
\titlespacing*{\subsection}{0pt}{2pc-1em}{17pt-1em}
\titleformat*{\subsubsection}{\normalsize\bfseries\slshape}
\titlespacing*{\subsubsection}{0pt}{\baselineskip}{0pt}
\titleformat{\paragraph}[runin]{\normalsize\itshape}{\theparagraph}{0pt}{}[.\enspace]
\titlespacing*{\paragraph}{0pt}{\baselineskip}{0pt}
  \titleformat{\subparagraph}[runin]{\relsize{-1}\uppercase}{\thesubparagraph}{0pt}{}[.\enspace]
\titlespacing*{\subparagraph}{0pt}{\baselineskip}{0pt}

\RequirePackage{enumitem}
\setlist{noitemsep}
\setlist[itemize]{leftmargin=12pt,topsep=16pt-1em}
\setlist[enumerate]
{itemindent=14pt,topsep=16.8pt-1em,
 label={\roman*.},labelsep=7pt,
 labelwidth=*,leftmargin=*,align=right}


% quotes and epigraphs
\RequirePackage{quoting}
\newcommand{\quotefont}{\fontsize{6.5pt}{9.5pt}\selectfont}
\quotingsetup{vskip=16pt-1em,indentfirst=false,font={quotefont},leftmargin=12pt,rightmargin=12pt}

\renewenvironment{quote}{\begin{quoting}}{\end{quoting}}

\renewenvironment{quotation}{\begin{quoting}}{\end{quoting}}

\newenvironment{epigraph}[1]
{\begin{quoting}[font={itshape,fontsize{7.5pt}{10.5pt}},vskip=15pt-1em,leftmargin=0pt,rightmargin=0pt]%
 \def\@quotesource{#1}}
{\par\vspace{\dimexpr 15pt-1em\relax}{\upshape---\@quotesource}\end{quoting}}

% Space above/below equations
\g@addto@macro\normalsize{%
  \setlength\abovedisplayskip{\baselineskip}%
  \setlength\belowdisplayskip{\baselineskip}%
  \setlength\abovedisplayshortskip{\baselineskip}%
  \setlength\belowdisplayshortskip{\baselineskip}%
}

\RequirePackage{stfloats}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
% Left-, right- and center-aligned auto-wrapping column types
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\newcolumntype{C}{>{\centering\arraybackslash}X}

\AtBeginEnvironment{tabularx}{%
  \fontsize{7pt}{10pt}\selectfont
}
\AtBeginEnvironment{tabular}{%
  \fontsize{7pt}{10pt}\selectfont
}

\setlength{\aboverulesep}{11pt-1em}
\setlength{\belowrulesep}{11pt-1em}
\setlength{\heavyrulewidth}{0.5pt}
\setlength{\lightrulewidth}{0.5pt}
\arrayrulecolor{jcolour}


\newcommand{\tnote}[1]{\textsuperscript{\textit{#1}}}
\newlist{tablenotes}{itemize}{1}
\setlist[tablenotes]{label={},labelsep=0pt,leftmargin=0pt,topsep={\dimexpr 12pt-1em\relax}}
\appto{\tablenotes}{\fontsize{6pt}{9pt}\selectfont}

\RequirePackage[tableposition=top]{caption}
\DeclareCaptionFont{figcaptionfont}{\fontsize{6pt}{9pt}\selectfont}
\DeclareCaptionFont{tabcaptionfont}{\fontsize{7pt}{9pt}\selectfont}
\captionsetup{justification=justified,singlelinecheck=false,labelfont={bf},labelsep=period}
\captionsetup*[figure]{font=figcaptionfont,skip=10pt}
\captionsetup*[table]{font=tabcaptionfont,skip=10pt-1em}

% Skips for floats
\setlength{\floatsep}{1pc+6pt}
\setlength{\intextsep}{1pc+6pt}
\setlength{\textfloatsep}{1pc+6pt}

% References
\if@numrefs
  \RequirePackage[numbers]{natbib}
  \bibliographystyle{../refs/vancouver-authoryear}
\fi
\if@alpharefs
    \RequirePackage{natbib}
    \bibliographystyle{dcu}
\fi
\setlength{\bibhang}{11pt}
\setlength{\bibsep}{0pt}
\renewcommand{\bibnumfmt}[1]{#1.}