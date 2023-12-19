using CSV
using DataFrames

function clean_word(data)
    x = []
    data = split(lowercase(data))
    for word in data
        if !isnothing(findfirst("#", word)) || !isnothing(findfirst("http", word)) || !isnothing(findfirst("@", word))
            continue
        else
            y = ""
            bank = ""
            prev = ' '
            for char in word
                if isletter(char)
                    if char != prev
                        prev = char
                        y = "$(y)$(bank)$(char)"
                        bank = ""
                    else
                        bank = "$(bank)$(char)"
                    end
                end
            end
            if y != ""
                push!(x, y)
            end
        end
    end
    return x
end

function clean_label(data)
    if !isnothing(findfirst("No", data))
        return "NoBully"
    else
        return "Bully"
    end
end

data = DataFrame(CSV.File(open("$(@__DIR__)/data.csv")))
data = data[:, 1:2]
data_new = []
label_new = []
word_count_bully = Dict()
data_size_bully = 0
word_count_no_bully = Dict()
data_size_no_bully = 0
row = size(data, 1)
@time for i in 1:row
    sentence = clean_word(data[i, 1])
    label = clean_label(data[i, 2])
    for word in sentence
        if label == "Bully"
            if isnothing(findfirst((==(word)), collect(keys(word_count_bully))))
                word_count_bully[word] = 0
            end
            word_count_bully[word] += 1
        else
            if isnothing(findfirst((==(word)), collect(keys(word_count_no_bully))))
                word_count_no_bully[word] = 0
            end
            word_count_no_bully[word] += 1
        end
    end
    if label == "Bully"
        global data_size_bully += 1
    else
        global data_size_no_bully += 1
    end
    push!(data_new, sentence)
    push!(label_new, label)
end
@time @simd for word in collect(keys(word_count_bully))
    word_count_bully[word] /= data_size_bully
end
@time @simd for word in collect(keys(word_count_no_bully))
    word_count_no_bully[word] /= data_size_no_bully
end

# convert vec{any} to array
label_guess = []
@time for sentence in data_new
    bully_point = 0
    no_bully_point = 0
    word_count_test = Dict()
    for word in sentence
        if isnothing(findfirst((==(word)), collect(keys(word_count_test))))
            word_count_test[word] = 0
        end
        word_count_test[word] += 1
    end
    for word in collect(keys(word_count_test))
        exist_no_bully = true
        exist_bully = true
        try
            word_count_bully[word]
        catch
            exist_bully = false
        end
        try
            word_count_no_bully[word]
        catch
            exist_no_bully = false
        end
        if exist_bully && exist_no_bully
            if abs(word_count_test[word] - word_count_no_bully[word]) > abs(word_count_test[word] - word_count_bully[word])
                no_bully_point += abs(word_count_test[word] - word_count_no_bully[word]) - abs(word_count_test[word] - word_count_bully[word])
            else
                bully_point += abs(word_count_test[word] - word_count_bully[word]) - abs(word_count_test[word] - word_count_no_bully[word])
            end
        elseif exist_bully
            no_bully_point += abs(word_count_test[word] - word_count_bully[word])
        elseif exist_no_bully
            bully_point += abs(word_count_test[word] - word_count_no_bully[word])
        end
    end
    if bully_point < no_bully_point
        push!(label_guess, "Bully")
    else
        push!(label_guess, "NoBully")
    end
end
guessed = 0
guessed_bully = 0
guessed_no_bully = 0
@time for i in eachindex(label_guess)
    if label_guess[i] == label_new[i]
        if label_guess[i] == "Bully"
            global guessed_bully += 1
        else
            global guessed_no_bully += 1
        end
        global guessed += 1
    end
end
accuracy = guessed / (data_size_bully + data_size_no_bully)

for i in eachindex(data_new)
    data_new[i] = join(data_new[i], " ")
end
data[!, :full_text] = data_new
data[!, :full_text] = map(x -> string("[", join(map(y -> "'" * y * "'", split(x, " ")), ", "), "]"), data[!, :full_text])
data[!, :Label] = label_new

println("\n========== Preprocessing Data ==========")
display(data)
CSV.write("preprocessed_data.csv", data)

println("\n========== Result ==========")
accuracy = guessed / (data_size_bully + data_size_no_bully)
print("Total Accuracy: ")
println(accuracy * 100, "%")
print("\nTotal Bully guessed: ")
println("$(guessed_bully)/$(data_size_bully)")
println("Percentage: $(guessed_bully/data_size_bully*100)%")
print("\nTotal NoBully guessed: ")
println("$(guessed_no_bully)/$(data_size_no_bully)")
println("Percentage: $(guessed_no_bully/data_size_no_bully*100)%")

# concate predicted label to data
data[!, :predicted_label] = label_guess
data[!, :Label] = map(x -> x == "Bully" ? "Bully" : "NoBully", data[!, :Label])

println("\n========== Prediction ==========")
display(data)

# save the new data and new label to csv
CSV.write("predicted_data.csv", data)