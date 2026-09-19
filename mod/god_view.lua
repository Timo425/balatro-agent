--- STEAMODDED / LOVELY BALATRO GOD-VIEW MOD ---
-- Exposes internal game state to the external Python sidecar
-- and receives simulated actions.

local json = require("json") -- Or bundled JSON serializer

local GodView = {
    port = 12345,
    host = "127.0.0.1"
}

function GodView.get_state()
    if not G or not G.GAME or not G.hand then
        return { status = "waiting_for_game" }
    end

    local hand_cards = {}
    if G.hand and G.hand.cards then
        for i, card in ipairs(G.hand.cards) do
            table.insert(hand_cards, {
                index = i,
                rank = card.base.value,
                suit = card.base.suit,
                id = card.ID
            })
        end
    end

    local jokers = {}
    if G.jokers and G.jokers.cards then
        for _, j in ipairs(G.jokers.cards) do
            table.insert(jokers, j.ability.name or "Unknown Joker")
        end
    end

    return {
        status = "in_round",
        ante = G.GAME.round_resets.ante or 1,
        blind_name = G.GAME.blind and G.GAME.blind.name or "Small Blind",
        target_chips = G.GAME.blind and G.GAME.blind.chips or 300,
        current_chips = G.GAME.chips or 0,
        hands_left = G.GAME.current_round.hands_left or 4,
        discards_left = G.GAME.current_round.discards_left or 3,
        money = G.GAME.dollars or 4,
        hand = hand_cards,
        jokers = jokers
    }
end

function GodView.execute_action(action)
    if action.type == "PLAY" then
        G.hand:unhighlight_all()
        for _, card_idx in ipairs(action.indices) do
            if G.hand.cards[card_idx] then
                G.hand.cards[card_idx]:highlight(true)
            end
        end
        G.FUNCS.play_cards_from_highlighted(nil)
    elseif action.type == "DISCARD" then
        G.hand:unhighlight_all()
        for _, card_idx in ipairs(action.indices) do
            if G.hand.cards[card_idx] then
                G.hand.cards[card_idx]:highlight(true)
            end
        end
        G.FUNCS.discard_cards_from_highlighted(nil)
    end
end

return GodView
