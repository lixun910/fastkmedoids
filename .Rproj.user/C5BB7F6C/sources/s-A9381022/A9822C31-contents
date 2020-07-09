#' An S4 class to represent the result of kmedoids clustering 
#' 
#' @slot cost The cost value of kmedoids clustering
#' @slot medoids The medoids of kmedoids clustering
#' @slot assignment The assignment of which cluster each observation belongs to of kmedoids clustering
setClass (
  # Class name
  "KmedoidsResult",
  
  # Defining slot type
  representation (
    cost = "numeric",
    medoids = "array",
    assignment = "array"
  ),
  
  # Initializing slots
  prototype = list(
    cost = as.numeric(0)
  )
)