library(shiny)
library(shinythemes)
setwd("C:/Users/abhishek.b.jaiswal/Desktop/HACK17/global")

shinyUI(fluidPage(theme = shinytheme("slate"),
  titlePanel(title="easy2insure"),
  br(),
  sidebarLayout(position="left",
                sidebarPanel(h3("Variables:"),
                             h5("Kindly select"),
                             selectInput("buying","Buying type",c("low", "med", "high", "vhigh")),
                             selectInput("maint","Maintenance level",c("low", "med", "high", "vhigh")),
                             selectInput("doors","Number of doors",c("2", "3", "4", "5more")),
                             selectInput("persons","Persons numbers",c("2", "4", "more")),
                             selectInput("lug_boot","Lug_boot space",c("small", "med", "big")),
                             selectInput("safety","Safty level",c("low", "med", "high")),
                             submitButton("Update")
                ),
                
                mainPanel(
                  h4("Your Car,Our insurance solution"),
                  p("Based on the car features,We offer Comprehensive and Partial Car Insurance"),
                  tabsetPanel(type="tab",
                              tabPanel("Info",br(),
                                       print("Easy2Insure Platform help the car owner and insurance company to check the insurnance coverage of  a given a car based on the conditions."),br(),
                                       print("It has inbuilt model to create the entire ecosystem around features of automobiles."),br(),
                                       print("Our model is based on classification and its using randomForest algorithm for analysis and predictions."),br(),
                                       print("Tool Used:"),br(),  
                                       print("1. R Studio"),br(),
                                       print("2. Shinyapps"),br(),br(),
                                       print("Car evaluation is being done based on 6 attribute like  buying price, maintenance charges, no of doors in car, persons capacity, lug boot space and safety features."),br(),
                                       print("Safety, cost, and luxury are important factors to consider in buying cars. These factors vary based on type, model, and manufacturer of the car."),
                                       print("However, these factors are so crucial in aspect of analyzing the cost of car insurance."),br(),br(),
                                       print("Key Features:"),br(),
                                       print("1.User friendly tool and easy to use."),br(),
                                       print("2.cars can be classified into four catagory: acceptable, anacceptable, good, very good."),br(),
                                       print("3.Based on classified category Insurance company can decide the insurance coverage.")),
                              
                              tabPanel("Result",textOutput("predRF"),br(),
                                       print("Cases:"),br(),
                                       print("1->Compplete cover"),br(),
                                       print("3->Partial cover")
                                       ),
                              tabPanel("Models",
                                       print("Naive Bayes Algorithm:         Accuracy:85.11 %"),br(),
                                       print("k-Nearest Neighbors Algorithm: Accuracy:90.11%"),br(),
                                       print("Decision Tree Algorithm:       Accuracy: 92.13% "),br(),
                                       print("CART Algorithm:                Accuracy:70.05%"),br(),
                                       print("C50 Algorithm:                 Accuracy:95.83%"),br(),
                                       print("Random Forest Algorithm:       Accuracy:98.21% ")
                                       ),
                              tabPanel("Sample",tableOutput("Data")),
                              tabPanel("Plot",plotOutput("Classplot")),
                              tabPanel("VarPlot",plotOutput("varplot")),
                              tabPanel("Data",textOutput("Buying"),textOutput("Maint"),textOutput("Doors"),textOutput("Persons"),textOutput("Lug_boot"),textOutput("Safety"))
                              
                  )))))