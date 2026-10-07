use Cwd 'getcwd';

my $repo_root = getcwd();

$ENV{'TEXMFVAR'} = "$repo_root/.texlive/texmf-var";
$ENV{'TEXMFCONFIG'} = "$repo_root/.texlive/texmf-config";

$pdf_mode = 1;
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$pdf_previewer = 'zathura %S';
