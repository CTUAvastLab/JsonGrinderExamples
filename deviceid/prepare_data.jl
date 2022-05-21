LABELS = [
    "HOME_AUTOMATION",
    "PRINTER",
    "VOICE_ASSISTANT",
    "GAME_CONSOLE",
    "SURVEILLANCE",
    "MEDIA_BOX",
    "IP_PHONE",
    "GENERIC_IOT",
    "NAS",
    "TV",
    "AUDIO",
    "PC",
    "MOBILE"
]
LABEL_MAPPING = Dict(reverse.(enumerate(LABELS)))

function load_train_data()
    x_train = JSON.parse.(readlines("dataset/train.json"))
    y_train = map(i -> LABEL_MAPPING[i["device_class"]], x_train)
    foreach(i -> delete!(i, "device_class"), x_train)
    foreach(i -> delete!(i, "device_id"), x_train)
    return x_train, y_train
end

function load_test_data()
    x_test = JSON.parse.(readlines("dataset/test.json"))
    solution = CSV.read("dataset/solution.csv", DataFrame) |> eachrow |> Dict
    y_test = map(i -> LABEL_MAPPING[solution[i["device_id"]]], x_test)
    foreach(i -> delete!(i, "device_id"), x_test)
    return x_test, y_test
end

function map_mac_vendor(data)
    mac_vendor = Dict([m => v for (m, v) in split.(readlines("mac_address_resolutions.txt"), "\t")]...)
    foreach(data) do x
        x["mac_vendor"] = mac_vendor[x["mac"]]
    end
    return data
end

function split_ip(data)
    foreach(data) do x
        if haskey(x, "ip")
            x["ip_split"] = Dict("ip_$i" => v for (i,v) in enumerate(String.(split(x["ip"], "."))))
        end
    end
    return data
end
