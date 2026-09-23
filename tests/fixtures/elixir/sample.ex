defmodule Payments.Refund do
  @moduledoc "Refund handling."

  def process(order) do
    validate(order)
    Payments.Ledger.record(order)
  end

  defp validate(order) do
    if order.total > 0, do: :ok, else: :error
  end
end
